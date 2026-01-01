-- MVP1 Inquiry Tables Migration
-- Run with: docker exec herenews-app python -c "..."
-- Or directly: psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f 001_create_inquiry_tables.sql

-- =============================================================================
-- INQUIRIES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS inquiries (
  id              VARCHAR(20) PRIMARY KEY,  -- inq_abc12345
  title           TEXT NOT NULL,
  description     TEXT DEFAULT '',
  status          VARCHAR(20) NOT NULL DEFAULT 'open',
  rigor_level     VARCHAR(1) NOT NULL DEFAULT 'B',
  schema_type     VARCHAR(30) NOT NULL DEFAULT 'boolean',
  schema_config   JSONB DEFAULT '{}',

  -- Scope constraints
  scope_entities  TEXT[] DEFAULT '{}',
  scope_keywords  TEXT[] DEFAULT '{}',
  scope_time_start TIMESTAMPTZ,
  scope_time_end  TIMESTAMPTZ,

  -- Current belief state
  posterior_map   JSONB,
  posterior_prob  DECIMAL(5,4) DEFAULT 0.0,
  entropy_bits    DECIMAL(6,3) DEFAULT 0.0,
  normalized_entropy DECIMAL(5,4) DEFAULT 0.0,
  credible_interval JSONB,

  -- Stakes
  total_stake     DECIMAL(12,2) DEFAULT 0.0,
  distributed     DECIMAL(12,2) DEFAULT 0.0,

  -- Counts (denormalized for performance)
  contribution_count INTEGER DEFAULT 0,
  open_tasks_count INTEGER DEFAULT 0,

  -- Metadata
  created_by      UUID REFERENCES users(user_id) ON DELETE SET NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at     TIMESTAMPTZ,

  -- Resolution tracking
  stable_since    TIMESTAMPTZ,

  CONSTRAINT valid_inquiry_status CHECK (status IN ('open', 'resolved', 'stale', 'closed')),
  CONSTRAINT valid_rigor_level CHECK (rigor_level IN ('A', 'B', 'C'))
);

-- Index for listing by status and stake
CREATE INDEX IF NOT EXISTS idx_inquiries_status_stake ON inquiries(status, total_stake DESC);
CREATE INDEX IF NOT EXISTS idx_inquiries_status_entropy ON inquiries(status, entropy_bits DESC);
CREATE INDEX IF NOT EXISTS idx_inquiries_created ON inquiries(created_at DESC);

-- Full text search on title
CREATE INDEX IF NOT EXISTS idx_inquiries_title_search ON inquiries USING GIN(to_tsvector('english', title));

-- =============================================================================
-- INQUIRY STAKES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS inquiry_stakes (
  id              SERIAL PRIMARY KEY,
  inquiry_id      VARCHAR(20) NOT NULL REFERENCES inquiries(id) ON DELETE CASCADE,
  user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  amount          DECIMAL(12,2) NOT NULL CHECK (amount > 0),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(inquiry_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_stakes_inquiry ON inquiry_stakes(inquiry_id);
CREATE INDEX IF NOT EXISTS idx_stakes_user ON inquiry_stakes(user_id);

-- =============================================================================
-- CONTRIBUTIONS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS contributions (
  id              VARCHAR(20) PRIMARY KEY,  -- contrib_abc123
  inquiry_id      VARCHAR(20) NOT NULL REFERENCES inquiries(id) ON DELETE CASCADE,
  user_id         UUID REFERENCES users(user_id) ON DELETE SET NULL,

  -- Content
  contribution_type VARCHAR(30) NOT NULL DEFAULT 'evidence',
  text            TEXT NOT NULL,
  source_url      TEXT,
  source_name     VARCHAR(255),

  -- Extracted value for typed inquiries
  extracted_value JSONB,
  observation_kind VARCHAR(20),

  -- Processing state
  processed       BOOLEAN DEFAULT false,
  claim_ids       TEXT[] DEFAULT '{}',

  -- Impact and rewards
  posterior_impact DECIMAL(5,4) DEFAULT 0.0,
  reward_earned   DECIMAL(12,2) DEFAULT 0.0,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT valid_contribution_type CHECK (
    contribution_type IN ('evidence', 'refutation', 'attribution', 'scope_correction', 'disambiguation')
  )
);

CREATE INDEX IF NOT EXISTS idx_contributions_inquiry ON contributions(inquiry_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_contributions_user ON contributions(user_id);
CREATE INDEX IF NOT EXISTS idx_contributions_impact ON contributions(inquiry_id, posterior_impact DESC);

-- =============================================================================
-- INQUIRY TASKS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS inquiry_tasks (
  id              VARCHAR(20) PRIMARY KEY,  -- task_abc123
  inquiry_id      VARCHAR(20) NOT NULL REFERENCES inquiries(id) ON DELETE CASCADE,

  task_type       VARCHAR(30) NOT NULL,
  description     TEXT NOT NULL,
  bounty          DECIMAL(12,2) DEFAULT 0.0,

  claimed_by      UUID REFERENCES users(user_id) ON DELETE SET NULL,
  claimed_at      TIMESTAMPTZ,
  completed       BOOLEAN DEFAULT false,
  completed_at    TIMESTAMPTZ,
  completion_contribution_id VARCHAR(20) REFERENCES contributions(id),

  meta_claim_id   VARCHAR(50),

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT valid_task_type CHECK (
    task_type IN ('need_primary_source', 'unresolved_conflict', 'single_source_only', 'high_entropy', 'stale')
  )
);

CREATE INDEX IF NOT EXISTS idx_tasks_inquiry ON inquiry_tasks(inquiry_id, completed);
CREATE INDEX IF NOT EXISTS idx_tasks_claimed ON inquiry_tasks(claimed_by) WHERE claimed_by IS NOT NULL;

-- =============================================================================
-- CREDIT TRANSACTIONS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS credit_transactions (
  id              SERIAL PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  amount          DECIMAL(12,2) NOT NULL,  -- positive=credit, negative=debit
  balance_after   DECIMAL(12,2) NOT NULL,

  transaction_type VARCHAR(30) NOT NULL,
  reference_type  VARCHAR(30),
  reference_id    VARCHAR(50),

  description     TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT valid_transaction_type CHECK (
    transaction_type IN ('stake', 'reward', 'purchase', 'refund', 'signup_bonus', 'admin_adjustment')
  )
);

CREATE INDEX IF NOT EXISTS idx_transactions_user ON credit_transactions(user_id, created_at DESC);

-- =============================================================================
-- TRIGGER: Update inquiry counts on contribution insert
-- =============================================================================
CREATE OR REPLACE FUNCTION update_inquiry_contribution_count()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE inquiries SET
      contribution_count = contribution_count + 1,
      updated_at = now()
    WHERE id = NEW.inquiry_id;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE inquiries SET
      contribution_count = GREATEST(0, contribution_count - 1),
      updated_at = now()
    WHERE id = OLD.inquiry_id;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_contribution_count ON contributions;
CREATE TRIGGER trigger_contribution_count
AFTER INSERT OR DELETE ON contributions
FOR EACH ROW EXECUTE FUNCTION update_inquiry_contribution_count();

-- =============================================================================
-- TRIGGER: Update inquiry stake total on stake change
-- =============================================================================
CREATE OR REPLACE FUNCTION update_inquiry_stake_total()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE inquiries SET
      total_stake = total_stake + NEW.amount,
      updated_at = now()
    WHERE id = NEW.inquiry_id;
  ELSIF TG_OP = 'UPDATE' THEN
    UPDATE inquiries SET
      total_stake = total_stake - OLD.amount + NEW.amount,
      updated_at = now()
    WHERE id = NEW.inquiry_id;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE inquiries SET
      total_stake = GREATEST(0, total_stake - OLD.amount),
      updated_at = now()
    WHERE id = OLD.inquiry_id;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_stake_total ON inquiry_stakes;
CREATE TRIGGER trigger_stake_total
AFTER INSERT OR UPDATE OR DELETE ON inquiry_stakes
FOR EACH ROW EXECUTE FUNCTION update_inquiry_stake_total();

-- =============================================================================
-- TRIGGER: Update inquiry task count
-- =============================================================================
CREATE OR REPLACE FUNCTION update_inquiry_task_count()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NOT NEW.completed THEN
      UPDATE inquiries SET open_tasks_count = open_tasks_count + 1, updated_at = now()
      WHERE id = NEW.inquiry_id;
    END IF;
  ELSIF TG_OP = 'UPDATE' THEN
    IF OLD.completed = false AND NEW.completed = true THEN
      UPDATE inquiries SET open_tasks_count = GREATEST(0, open_tasks_count - 1), updated_at = now()
      WHERE id = NEW.inquiry_id;
    ELSIF OLD.completed = true AND NEW.completed = false THEN
      UPDATE inquiries SET open_tasks_count = open_tasks_count + 1, updated_at = now()
      WHERE id = NEW.inquiry_id;
    END IF;
  ELSIF TG_OP = 'DELETE' THEN
    IF NOT OLD.completed THEN
      UPDATE inquiries SET open_tasks_count = GREATEST(0, open_tasks_count - 1), updated_at = now()
      WHERE id = OLD.inquiry_id;
    END IF;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_task_count ON inquiry_tasks;
CREATE TRIGGER trigger_task_count
AFTER INSERT OR UPDATE OR DELETE ON inquiry_tasks
FOR EACH ROW EXECUTE FUNCTION update_inquiry_task_count();

-- =============================================================================
-- COMMENTS: Add helpful notes
-- =============================================================================
COMMENT ON TABLE inquiries IS 'User-scoped questions with typed targets and live evidence state';
COMMENT ON TABLE inquiry_stakes IS 'User stakes/bounties on inquiries';
COMMENT ON TABLE contributions IS 'User submissions to inquiries (evidence, refutations, etc.)';
COMMENT ON TABLE inquiry_tasks IS 'System-generated tasks from meta-claims';
COMMENT ON TABLE credit_transactions IS 'Audit log for all credit movements';

-- =============================================================================
-- PRINT SUCCESS
-- =============================================================================
DO $$
BEGIN
  RAISE NOTICE 'MVP1 Inquiry tables created successfully!';
END $$;

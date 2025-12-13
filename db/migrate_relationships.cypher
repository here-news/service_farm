// Migration: Rename relationships to organism-based naming
// Run this ONCE to migrate existing data

// 1. Page -[:CONTAINS]-> Claim  →  Page -[:EMITS]-> Claim
MATCH (p:Page)-[r:CONTAINS]->(c:Claim)
CALL {
    WITH p, r, c
    CREATE (p)-[r2:EMITS]->(c)
    SET r2 = properties(r)
    DELETE r
} IN TRANSACTIONS OF 1000 ROWS;

// 2. Event -[:SUPPORTS]-> Claim  →  Event -[:INTAKES]-> Claim
MATCH (e:Event)-[r:SUPPORTS]->(c:Claim)
CALL {
    WITH e, r, c
    CREATE (e)-[r2:INTAKES]->(c)
    SET r2 = properties(r)
    DELETE r
} IN TRANSACTIONS OF 1000 ROWS;

// 3. Event -[:CONTAINS]-> Event  →  Event -[:SPAWNS]-> Event
MATCH (parent:Event)-[r:CONTAINS]->(child:Event)
CALL {
    WITH parent, r, child
    CREATE (parent)-[r2:SPAWNS]->(child)
    SET r2 = properties(r)
    DELETE r
} IN TRANSACTIONS OF 1000 ROWS;

// 4. Page -[:HAS_CLAIM]-> Claim  →  Page -[:EMITS]-> Claim (legacy cleanup)
MATCH (p:Page)-[r:HAS_CLAIM]->(c:Claim)
CALL {
    WITH p, r, c
    CREATE (p)-[r2:EMITS]->(c)
    SET r2 = properties(r)
    DELETE r
} IN TRANSACTIONS OF 1000 ROWS;

// Verify migration
MATCH ()-[r:CONTAINS|SUPPORTS|HAS_CLAIM]->()
RETURN type(r) as old_type, count(*) as remaining
ORDER BY old_type;

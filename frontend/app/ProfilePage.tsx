import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { User } from './types/user'

interface Transaction {
  id: string
  amount: number
  balance_after: number
  transaction_type: string
  reference_type?: string
  reference_id?: string
  created_at: string
}

interface Stake {
  inquiry_id: string
  inquiry_title: string
  inquiry_status: string
  stake_amount: number
  total_stake: number
  posterior_prob: number
  staked_at: string
}

interface Contribution {
  id: string
  inquiry_id: string
  inquiry_title: string
  type: string
  text: string
  impact: number
  reward_earned: number
  created_at: string
}

interface ProfileData {
  user_id: string
  email: string
  name: string
  picture_url?: string
  credits_balance: number
  reputation: number
  subscription_tier: string
  created_at: string
  stats: {
    total_contributions: number
    total_stakes: number
    total_rewards: number
  }
}

type TabType = 'overview' | 'stakes' | 'contributions' | 'transactions'

function ProfilePage() {
  const navigate = useNavigate()
  const [user, setUser] = useState<User | null>(null)
  const [profile, setProfile] = useState<ProfileData | null>(null)
  const [stakes, setStakes] = useState<Stake[]>([])
  const [contributions, setContributions] = useState<Contribution[]>([])
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [activeTab, setActiveTab] = useState<TabType>('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    checkAuthAndLoad()
  }, [])

  const checkAuthAndLoad = async () => {
    try {
      // Check auth first
      const authRes = await fetch('/api/auth/status')
      const authData = await authRes.json()

      if (!authData.authenticated) {
        navigate('/')
        return
      }

      setUser(authData.user)

      // Load profile data
      await Promise.all([
        loadProfile(),
        loadStakes(),
        loadContributions(),
        loadTransactions()
      ])
    } catch (err) {
      setError('Failed to load profile')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadProfile = async () => {
    try {
      const res = await fetch('/api/user/profile')
      if (res.ok) {
        const data = await res.json()
        setProfile(data)
      }
    } catch (err) {
      console.error('Failed to load profile:', err)
    }
  }

  const loadStakes = async () => {
    try {
      const res = await fetch('/api/user/stakes')
      if (res.ok) {
        const data = await res.json()
        setStakes(data.stakes || [])
      }
    } catch (err) {
      console.error('Failed to load stakes:', err)
    }
  }

  const loadContributions = async () => {
    try {
      const res = await fetch('/api/user/contributions?limit=50')
      if (res.ok) {
        const data = await res.json()
        setContributions(data.contributions || [])
      }
    } catch (err) {
      console.error('Failed to load contributions:', err)
    }
  }

  const loadTransactions = async () => {
    try {
      const res = await fetch('/api/user/transactions?limit=50')
      if (res.ok) {
        const data = await res.json()
        setTransactions(data.transactions || [])
      }
    } catch (err) {
      console.error('Failed to load transactions:', err)
    }
  }

  const formatCredits = (amount: number) => {
    return new Intl.NumberFormat().format(amount)
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  const formatTime = (dateStr: string) => {
    return new Date(dateStr).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-slate-500">Loading profile...</div>
      </div>
    )
  }

  if (error || !user) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <div className="text-red-500">{error || 'Please sign in to view your profile'}</div>
        <Link to="/" className="text-indigo-600 hover:underline">Return home</Link>
      </div>
    )
  }

  const totalStaked = stakes.reduce((sum, s) => sum + s.stake_amount, 0)
  const totalRewards = contributions.reduce((sum, c) => sum + c.reward_earned, 0)
  const totalImpact = contributions.reduce((sum, c) => sum + c.impact, 0)

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Profile Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 mb-6">
        <div className="flex items-start gap-6">
          {/* Avatar */}
          {user.picture ? (
            <img
              src={user.picture}
              alt={user.name}
              className="w-20 h-20 rounded-full border-4 border-white shadow-lg"
            />
          ) : (
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white text-2xl font-bold shadow-lg">
              {user.name.charAt(0).toUpperCase()}
            </div>
          )}

          {/* User Info */}
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-slate-900">{user.name}</h1>
            <p className="text-slate-500">{user.email}</p>
            {profile?.created_at && (
              <p className="text-sm text-slate-400 mt-1">
                Member since {formatDate(profile.created_at)}
              </p>
            )}
          </div>

          {/* Credits Card */}
          <div className="bg-gradient-to-br from-amber-50 to-yellow-50 border border-amber-200 rounded-xl p-4 min-w-[180px]">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-5 h-5 text-amber-600" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <span className="text-sm font-medium text-amber-700">Credits</span>
            </div>
            <div className="text-3xl font-bold text-amber-800">
              {formatCredits(profile?.credits_balance || user.credits || 0)}
            </div>
            <div className="text-xs text-amber-600 mt-1">
              ≈ ${((profile?.credits_balance || 0) / 100).toFixed(2)}
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-100">
          <div className="text-center">
            <div className="text-2xl font-bold text-slate-900">{contributions.length}</div>
            <div className="text-sm text-slate-500">Contributions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-slate-900">{stakes.length}</div>
            <div className="text-sm text-slate-500">Active Stakes</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">+{formatCredits(Math.round(totalRewards))}</div>
            <div className="text-sm text-slate-500">Total Earned</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-indigo-600">{totalImpact.toFixed(2)}</div>
            <div className="text-sm text-slate-500">Impact Score</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 bg-slate-100 p-1 rounded-lg w-fit">
        {(['overview', 'stakes', 'contributions', 'transactions'] as TabType[]).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition capitalize ${
              activeTab === tab
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-xl border border-slate-200">
        {activeTab === 'overview' && (
          <div className="p-6 space-y-6">
            {/* Quick Stats */}
            <div>
              <h3 className="text-lg font-semibold text-slate-900 mb-4">Your Activity</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-50 rounded-lg p-4">
                  <div className="text-sm text-slate-500 mb-1">Total Staked</div>
                  <div className="text-xl font-bold text-slate-900">{formatCredits(totalStaked)} credits</div>
                  <div className="text-xs text-slate-400">${(totalStaked / 100).toFixed(2)}</div>
                </div>
                <div className="bg-slate-50 rounded-lg p-4">
                  <div className="text-sm text-slate-500 mb-1">Reputation</div>
                  <div className="text-xl font-bold text-slate-900">{profile?.reputation || 0}</div>
                  <div className="text-xs text-slate-400">Based on accuracy</div>
                </div>
              </div>
            </div>

            {/* Recent Stakes */}
            {stakes.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-slate-900 mb-3">Recent Stakes</h3>
                <div className="space-y-2">
                  {stakes.slice(0, 3).map(stake => (
                    <Link
                      key={stake.inquiry_id}
                      to={`/inquiry/${stake.inquiry_id}`}
                      className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition"
                    >
                      <div>
                        <div className="font-medium text-slate-900">{stake.inquiry_title}</div>
                        <div className="text-sm text-slate-500">
                          {Math.round(stake.posterior_prob * 100)}% confidence
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-amber-600">{formatCredits(stake.stake_amount)}</div>
                        <div className="text-xs text-slate-400">{formatDate(stake.staked_at)}</div>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Recent Contributions */}
            {contributions.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-slate-900 mb-3">Recent Contributions</h3>
                <div className="space-y-2">
                  {contributions.slice(0, 3).map(contrib => (
                    <Link
                      key={contrib.id}
                      to={`/inquiry/${contrib.inquiry_id}`}
                      className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition"
                    >
                      <div>
                        <div className="font-medium text-slate-900">{contrib.inquiry_title}</div>
                        <div className="text-sm text-slate-500 truncate max-w-md">{contrib.text}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-green-600">+{Math.round(contrib.reward_earned)}</div>
                        <div className="text-xs text-slate-400">Impact: {contrib.impact.toFixed(2)}</div>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {stakes.length === 0 && contributions.length === 0 && (
              <div className="text-center py-8 text-slate-500">
                <p className="mb-4">You haven't made any contributions or stakes yet.</p>
                <Link
                  to="/"
                  className="inline-flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-medium"
                >
                  Explore inquiries to get started
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Link>
              </div>
            )}
          </div>
        )}

        {activeTab === 'stakes' && (
          <div className="divide-y divide-slate-100">
            {stakes.length === 0 ? (
              <div className="p-8 text-center text-slate-500">
                No stakes yet. Stake on inquiries to signal importance and earn rewards.
              </div>
            ) : (
              stakes.map(stake => (
                <Link
                  key={stake.inquiry_id}
                  to={`/inquiry/${stake.inquiry_id}`}
                  className="flex items-center justify-between p-4 hover:bg-slate-50 transition"
                >
                  <div className="flex-1">
                    <div className="font-medium text-slate-900">{stake.inquiry_title}</div>
                    <div className="flex items-center gap-3 mt-1">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        stake.inquiry_status === 'open'
                          ? 'bg-green-100 text-green-700'
                          : 'bg-slate-100 text-slate-600'
                      }`}>
                        {stake.inquiry_status}
                      </span>
                      <span className="text-sm text-slate-500">
                        Total pool: {formatCredits(stake.total_stake)}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-amber-600">{formatCredits(stake.stake_amount)}</div>
                    <div className="text-xs text-slate-400">{formatDate(stake.staked_at)}</div>
                  </div>
                </Link>
              ))
            )}
          </div>
        )}

        {activeTab === 'contributions' && (
          <div className="divide-y divide-slate-100">
            {contributions.length === 0 ? (
              <div className="p-8 text-center text-slate-500">
                No contributions yet. Help resolve inquiries to earn credits.
              </div>
            ) : (
              contributions.map(contrib => (
                <Link
                  key={contrib.id}
                  to={`/inquiry/${contrib.inquiry_id}`}
                  className="block p-4 hover:bg-slate-50 transition"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="font-medium text-slate-900">{contrib.inquiry_title}</div>
                      <div className="text-sm text-slate-600 mt-1">{contrib.text}</div>
                      <div className="flex items-center gap-3 mt-2">
                        <span className="text-xs px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full capitalize">
                          {contrib.type.replace('_', ' ')}
                        </span>
                        <span className="text-xs text-slate-400">{formatTime(contrib.created_at)}</span>
                      </div>
                    </div>
                    <div className="text-right ml-4">
                      <div className="font-bold text-green-600">+{Math.round(contrib.reward_earned)}</div>
                      <div className="text-xs text-slate-500">Impact: {contrib.impact.toFixed(3)}</div>
                    </div>
                  </div>
                </Link>
              ))
            )}
          </div>
        )}

        {activeTab === 'transactions' && (
          <div className="divide-y divide-slate-100">
            {transactions.length === 0 ? (
              <div className="p-8 text-center text-slate-500">
                No transactions yet.
              </div>
            ) : (
              transactions.map((tx, i) => (
                <div key={i} className="flex items-center justify-between p-4">
                  <div>
                    <div className="font-medium text-slate-900 capitalize">
                      {tx.transaction_type.replace('_', ' ')}
                    </div>
                    {tx.reference_type && (
                      <div className="text-sm text-slate-500">
                        {tx.reference_type}: {tx.reference_id?.slice(0, 8)}...
                      </div>
                    )}
                    <div className="text-xs text-slate-400 mt-1">{formatTime(tx.created_at)}</div>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold ${tx.amount >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {tx.amount >= 0 ? '+' : ''}{formatCredits(tx.amount)}
                    </div>
                    <div className="text-xs text-slate-400">
                      Balance: {formatCredits(tx.balance_after)}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ProfilePage

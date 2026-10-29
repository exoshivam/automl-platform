import { create } from 'zustand';
import type { User } from '@/api/types';

/** Access token lives in memory only (ACC-2); the refresh token is an httpOnly cookie. */
interface AuthState {
  accessToken: string | null;
  user: User | null;
  status: 'unknown' | 'authenticated' | 'anonymous';
  setSession: (token: string, user: User) => void;
  clear: () => void;
}

export const useAuth = create<AuthState>(set => ({
  accessToken: null,
  user: null,
  status: 'unknown',
  setSession: (accessToken, user) => set({ accessToken, user, status: 'authenticated' }),
  clear: () => set({ accessToken: null, user: null, status: 'anonymous' }),
}));

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type Theme = 'light' | 'dark' | 'system';

interface UiState {
  theme: Theme;
  /** Selected workspace (org id). Not sensitive, so it persists across reloads. */
  orgId: string | null;
  setTheme: (t: Theme) => void;
  setOrgId: (id: string) => void;
}

export function applyTheme(theme: Theme) {
  const dark = theme === 'dark' || (theme === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', dark);
  try {
    localStorage.setItem('automl.theme', theme);
  } catch {
    /* ignore */
  }
}

export const useUi = create<UiState>()(
  persist(
    set => ({
      theme: 'system',
      orgId: null,
      setTheme: theme => {
        applyTheme(theme);
        set({ theme });
      },
      setOrgId: orgId => set({ orgId }),
    }),
    { name: 'automl.ui', partialize: s => ({ theme: s.theme, orgId: s.orgId }) },
  ),
);

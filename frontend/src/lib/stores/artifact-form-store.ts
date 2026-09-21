import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import type { ArtifactKind } from '@/lib/types/artifacts'

const KINDS: readonly ArtifactKind[] = ['report', 'deck']

/**
 * Coerce anything that came from the user or from storage to a valid kind.
 *
 * A persisted value outlives the code that wrote it, so it is never trusted:
 * if the set of valid kinds ever changes, an old entry must not reach the
 * Select, where it would render as an empty option.
 */
function sanitizeKind(kind: unknown): ArtifactKind {
  return KINDS.includes(kind as ArtifactKind) ? (kind as ArtifactKind) : 'report'
}

interface ArtifactFormState {
  kind: ArtifactKind
  setKind: (kind: ArtifactKind) => void
}

/**
 * Preferences of the artifact generation form, kept between visits.
 *
 * Only the kind is remembered: the language follows the UI language and the
 * other fields are per-run choices. The kind is the one selection whose
 * consequence (a deck is written as bullets and carries diagrams) is not
 * visible in the form itself, so making the user redo it every time is the
 * difference between getting diagrams or not.
 */
export const useArtifactFormStore = create<ArtifactFormState>()(
  persist(
    (set) => ({
      kind: 'report',
      setKind: (kind) => set({ kind: sanitizeKind(kind) }),
    }),
    {
      name: 'artifact-form-storage',
      merge: (persisted, current) => {
        const stored = persisted as Partial<ArtifactFormState> | undefined
        return { ...current, kind: sanitizeKind(stored?.kind) }
      },
    }
  )
)

export type ArtifactKind = 'report' | 'deck'

export type ArtifactFormat = 'md' | 'html' | 'docx' | 'pptx'

/** One generated report or deck, plus the status of its generating job. */
export interface Artifact {
  id: string
  notebook_id: string
  title: string
  kind: ArtifactKind
  formats: ArtifactFormat[]
  language: string
  sections: number
  output_path?: string | null
  note_id?: string | null
  created?: string | null
  job_status?: string | null
  error_message?: string | null
  download_url?: string | null
  files: string[]
}

export interface GenerateArtifactRequest {
  notebook_id: string
  kind?: ArtifactKind
  formats?: ArtifactFormat[]
  language?: string
  title?: string
  instructions?: string
  sections?: number
}

export interface GenerateArtifactResponse {
  command_id: string
  artifact_id: string
  status: string
}

/** Job statuses that mean "the worker is still working on this". */
export const ACTIVE_ARTIFACT_STATUSES = [
  'pending',
  'running',
  'processing',
  'submitted',
  'queued',
  'new',
] as const

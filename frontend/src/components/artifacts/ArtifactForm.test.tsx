import { describe, expect, it } from 'vitest'

import { defaultArtifactLanguage } from './ArtifactForm'

describe('defaultArtifactLanguage', () => {
  it('maps a regional UI language to its base code', () => {
    expect(defaultArtifactLanguage('it-IT')).toBe('it')
    expect(defaultArtifactLanguage('en-US')).toBe('en')
    expect(defaultArtifactLanguage('zh-CN')).toBe('zh')
    expect(defaultArtifactLanguage('ru-RU')).toBe('ru')
  })

  it('accepts a bare language code', () => {
    expect(defaultArtifactLanguage('fr')).toBe('fr')
  })

  it('falls back to English for unsupported or empty languages', () => {
    expect(defaultArtifactLanguage('')).toBe('en')
    expect(defaultArtifactLanguage('xx-YY')).toBe('en')
  })
})

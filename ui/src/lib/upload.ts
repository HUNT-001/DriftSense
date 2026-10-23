/** Client-side mirror of the service's upload constraints (POST /v1/uploads). */
export const ALLOWED_CONTENT_TYPES = ['image/png', 'image/tiff'] as const;
export type UploadContentType = (typeof ALLOWED_CONTENT_TYPES)[number];
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

export interface FileLike {
  name: string;
  type: string;
  size: number;
}

/** Browsers sometimes report an empty MIME type (notably for TIFF); fall back to the extension. */
export function contentTypeFor(file: FileLike): string {
  const t = (file.type || '').toLowerCase();
  if (t === 'image/tif') return 'image/tiff';
  if (t) return t;
  const ext = file.name.toLowerCase().split('.').pop() ?? '';
  if (ext === 'png') return 'image/png';
  if (ext === 'tif' || ext === 'tiff') return 'image/tiff';
  return '';
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Returns a user-facing error message, or null when the file is acceptable. */
export function validateUploadFile(file: FileLike | null | undefined): string | null {
  if (!file) return 'Choose a file.';
  const ct = contentTypeFor(file);
  if (!(ALLOWED_CONTENT_TYPES as readonly string[]).includes(ct)) {
    return `"${file.name}" is not a PNG or TIFF image.`;
  }
  if (file.size <= 0) return `"${file.name}" is empty.`;
  if (file.size > MAX_UPLOAD_BYTES) {
    return `"${file.name}" is ${formatBytes(file.size)}; the limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`;
  }
  return null;
}

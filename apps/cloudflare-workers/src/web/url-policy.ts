import type {
  WebPageReadFailure,
  WebPageReaderErrorCode,
} from '@echo-chamber/core/ports/web-page-reader';

interface AdmittedWebUrl {
  success: true;
  url: URL;
}

export type WebUrlAdmission = AdmittedWebUrl | WebPageReadFailure;

const MAX_URL_LENGTH = 4_096;
const ALLOWED_PROTOCOLS = new Set(['http:', 'https:']);
const INTERNAL_HOST_SUFFIXES = [
  '.internal',
  '.localhost',
  '.localdomain',
  '.local',
  '.home',
  '.lan',
  '.test',
  '.invalid',
  '.example',
  '.onion',
] as const;
const DOCUMENTATION_HOSTS = new Set([
  'example',
  'example.com',
  'example.net',
  'example.org',
]);
const CREDENTIAL_QUERY_KEYS = new Set([
  'access-key',
  'access-token',
  'api-key',
  'apikey',
  'auth',
  'authorization',
  'awsaccesskeyid',
  'credential',
  'googleaccessid',
  'key',
  'password',
  'secret',
  'sig',
  'signature',
  'token',
]);

function failure(
  code: WebPageReaderErrorCode,
  error: string
): WebPageReadFailure {
  return {
    success: false,
    code,
    error,
    retryable: false,
  };
}

function hasCredentialQuery(url: URL): boolean {
  for (const key of url.searchParams.keys()) {
    const normalizedKey = key.toLowerCase().replaceAll('_', '-');
    if (
      CREDENTIAL_QUERY_KEYS.has(normalizedKey) ||
      normalizedKey.startsWith('x-amz-') ||
      normalizedKey.startsWith('x-goog-')
    ) {
      return true;
    }
  }

  return false;
}

function parseIpv4(hostname: string): number | null {
  const parts = hostname.split('.');
  if (parts.length !== 4) {
    return null;
  }

  let value = 0;
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) {
      return null;
    }
    const octet = Number(part);
    if (!Number.isInteger(octet) || octet < 0 || octet > 255) {
      return null;
    }
    value = (value * 256 + octet) >>> 0;
  }

  return value;
}

function isIpv4InCidr(
  address: number,
  network: number,
  prefix: number
): boolean {
  const mask = (0xffffffff << (32 - prefix)) >>> 0;
  return (address & mask) >>> 0 === (network & mask) >>> 0;
}

function isNonPublicIpv4(address: number): boolean {
  const rejectedCidrs: readonly [number, number][] = [
    [0x00000000, 8],
    [0x0a000000, 8],
    [0x64400000, 10],
    [0x7f000000, 8],
    [0xa9fe0000, 16],
    [0xac100000, 12],
    [0xc0000000, 24],
    [0xc0000200, 24],
    [0xc0a80000, 16],
    [0xc6120000, 15],
    [0xc6336400, 24],
    [0xcb007100, 24],
    [0xe0000000, 4],
    [0xf0000000, 4],
  ];

  return rejectedCidrs.some(([network, prefix]) =>
    isIpv4InCidr(address, network, prefix)
  );
}

function parseIpv6Half(half: string): number[] | null {
  if (half === '') {
    return [];
  }

  const parsed: number[] = [];
  for (const group of half.split(':')) {
    if (!/^[\da-f]{1,4}$/i.test(group)) {
      return null;
    }
    parsed.push(Number.parseInt(group, 16));
  }
  return parsed;
}

function expandIpv6Halves(halves: readonly string[]): number[] | null {
  if (halves.length > 2) {
    return null;
  }

  const [leftText = '', rightText = ''] = halves;
  const left = parseIpv6Half(leftText);
  const right = parseIpv6Half(rightText);
  if (left === null || right === null) {
    return null;
  }

  if (halves.length === 1) {
    return left.length === 8 ? left : null;
  }

  const missingGroups = 8 - left.length - right.length;
  if (missingGroups < 1) {
    return null;
  }

  return [...left, ...Array.from({ length: missingGroups }, () => 0), ...right];
}

function parseIpv6(hostname: string): number[] | null {
  const hasBrackets = hostname.startsWith('[') && hostname.endsWith(']');
  const unwrapped = hasBrackets ? hostname.slice(1, -1) : hostname;
  return unwrapped.includes(':')
    ? expandIpv6Halves(unwrapped.split('::'))
    : null;
}

function isAllZero(groups: readonly number[], endExclusive: number): boolean {
  return groups.slice(0, endExclusive).every((group) => group === 0);
}

function getIpv4MappedAddress(groups: readonly number[]): number | null {
  if (!isAllZero(groups, 5) || groups[5] !== 0xffff) {
    return null;
  }

  return (((groups[6] ?? 0) << 16) | (groups[7] ?? 0)) >>> 0;
}

function isBasicNonPublicIpv6(groups: readonly number[]): boolean {
  return (
    groups.every((group) => group === 0) ||
    (isAllZero(groups, 7) && groups[7] === 1) ||
    isAllZero(groups, 6)
  );
}

function isLocalOrMulticastIpv6(groups: readonly number[]): boolean {
  const first = groups[0] ?? 0;
  return (
    (first & 0xfe00) === 0xfc00 ||
    (first & 0xffc0) === 0xfe80 ||
    (first & 0xff00) === 0xff00
  );
}

function isDiscardOnlyIpv6(groups: readonly number[]): boolean {
  return (groups[0] ?? 0) === 0x0100 && isAllZero(groups.slice(1), 3);
}

function isTranslationIpv6(groups: readonly number[]): boolean {
  return (
    (groups[0] ?? 0) === 0x0064 && groups[1] === 0xff9b && groups[2] === 0x0001
  );
}

function isDocumentationOrBenchmarkIpv6(groups: readonly number[]): boolean {
  return (
    (groups[0] ?? 0) === 0x2001 &&
    (groups[1] === 0x0002 || groups[1] === 0x0db8)
  );
}

function isSpecialRegistryIpv6(groups: readonly number[]): boolean {
  const first = groups[0] ?? 0;
  const second = groups[1] ?? 0;
  return (
    (first === 0x2001 &&
      ((second & 0xfff0) === 0x0010 || (second & 0xfff0) === 0x0020)) ||
    (first === 0x3fff && (second & 0xf000) === 0)
  );
}

function isSpecialUseIpv6(groups: readonly number[]): boolean {
  return [
    isBasicNonPublicIpv6(groups),
    isLocalOrMulticastIpv6(groups),
    isDiscardOnlyIpv6(groups),
    isTranslationIpv6(groups),
    isDocumentationOrBenchmarkIpv6(groups),
    isSpecialRegistryIpv6(groups),
  ].some(Boolean);
}

function isNonPublicIpv6(groups: readonly number[]): boolean {
  const mappedIpv4 = getIpv4MappedAddress(groups);
  if (mappedIpv4 !== null) {
    return isNonPublicIpv4(mappedIpv4);
  }

  return isSpecialUseIpv6(groups);
}

function isNonPublicHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/\.$/, '');
  const ipv4 = parseIpv4(normalized);
  if (ipv4 !== null) {
    return isNonPublicIpv4(ipv4);
  }
  const ipv6 = parseIpv6(normalized);
  if (ipv6 !== null) {
    return isNonPublicIpv6(ipv6);
  }

  return (
    normalized === 'localhost' ||
    DOCUMENTATION_HOSTS.has(normalized) ||
    INTERNAL_HOST_SUFFIXES.some(
      (suffix) => normalized === suffix.slice(1) || normalized.endsWith(suffix)
    )
  );
}

function parseCandidateUrl(rawUrl: string): URL | WebPageReadFailure {
  if (
    rawUrl.length < 1 ||
    rawUrl.length > MAX_URL_LENGTH ||
    rawUrl.trim() !== rawUrl
  ) {
    return failure('invalid_url', 'The URL is invalid.');
  }

  try {
    return new URL(rawUrl);
  } catch {
    return failure('invalid_url', 'The URL is invalid.');
  }
}

function validateCandidateUrl(url: URL): WebPageReadFailure | undefined {
  if (!ALLOWED_PROTOCOLS.has(url.protocol) || url.port !== '') {
    return failure('invalid_url', 'The URL is invalid.');
  }
  if (url.username !== '' || url.password !== '' || hasCredentialQuery(url)) {
    return failure(
      'sensitive_url',
      'The URL contains credentials or credential-like parameters.'
    );
  }
  if (isNonPublicHostname(url.hostname)) {
    return failure(
      'destination_not_public',
      'The URL does not identify an admitted public Web page.'
    );
  }

  return undefined;
}

/**
 * モデル指定URLを公開Web取得ポリシーへadmitする。
 * DNS解決後の宛先制御はCloudflareのplatform egress boundaryへ委ねる。
 */
export function admitPublicWebUrl(rawUrl: string): WebUrlAdmission {
  const parsed = parseCandidateUrl(rawUrl);
  if (!(parsed instanceof URL)) {
    return parsed;
  }
  const validationFailure = validateCandidateUrl(parsed);
  if (validationFailure !== undefined) {
    return validationFailure;
  }

  parsed.hash = '';
  if (!parsed.hostname.startsWith('[')) {
    parsed.hostname = parsed.hostname.replace(/\.$/, '');
  }

  return { success: true, url: parsed };
}

/**
 * redirect先を現在URLから解決し、初回と同じ公開URLポリシーを適用する。
 */
export function admitWebRedirect(
  currentUrl: URL,
  location: string
): WebUrlAdmission {
  let resolved: URL;
  try {
    resolved = new URL(location, currentUrl);
  } catch {
    return failure(
      'redirect_not_allowed',
      'The redirect destination is not allowed.'
    );
  }

  if (currentUrl.protocol === 'https:' && resolved.protocol !== 'https:') {
    return failure(
      'redirect_not_allowed',
      'The redirect destination is not allowed.'
    );
  }

  const admitted = admitPublicWebUrl(resolved.href);
  if (!admitted.success) {
    return failure(
      'redirect_not_allowed',
      'The redirect destination is not allowed.'
    );
  }

  return admitted;
}

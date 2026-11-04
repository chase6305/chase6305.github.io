"""Resolve local web-manifest icons without fetching external resources."""
import json
from urllib.parse import unquote, urljoin, urlsplit

_EDGE_URL_SPACE = ''.join(chr(value) for value in range(0x21))
_ASCII_NEWLINES = {ord(character): None for character in '\t\n\r'}
_DEFAULT_PORTS = {'http': 80, 'https': 443}


def _url_input(value):
    # URL input cleanup is ASCII-only: encoded spaces and NBSP filenames count.
    return value.strip(_EDGE_URL_SPACE).translate(_ASCII_NEWLINES)


def _http_separators(value, inherited_scheme=''):
    scheme = urlsplit(value).scheme or inherited_scheme
    if scheme in _DEFAULT_PORTS:
        end = min((index for marker in ('?', '#')
                   if (index := value.find(marker)) >= 0), default=len(value))
        value = value[:end].replace('\\', '/') + value[end:]
    return value


def resolve_reference_url(document_url, reference):
    """Resolve the input/separator forms used by these local resource checks.

    HTTP(S) treats path backslashes as separators; queries/fragments retain
    theirs. This small adapter does not implement a complete URL parser.
    """
    document_url = _http_separators(_url_input(document_url))
    reference = _http_separators(_url_input(reference), urlsplit(document_url).scheme)
    return urljoin(document_url, reference)


def _origin(parts):
    if parts.scheme not in _DEFAULT_PORTS or parts.hostname is None:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    return (parts.scheme, parts.hostname.lower(),
            _DEFAULT_PORTS[parts.scheme] if port is None else port)


def local_reference(public, base, document_url, reference):
    """Map a same-origin URL inside the published prefix to its build file."""
    public = public.resolve()
    site = urlsplit(resolve_reference_url(base, ''))
    parts = urlsplit(resolve_reference_url(document_url, reference))
    origin = _origin(site)
    if origin is None or _origin(parts) != origin:
        return None
    prefix = unquote(site.path)
    path = unquote(parts.path)
    if not path.startswith(prefix):
        return None
    local = public / path[len(prefix):]
    if not local.resolve().is_relative_to(public):
        raise ValueError('Asset is outside the production build: ' + str(local))
    return local


def manifest_icon_files(public, base, manifest_url):
    """Yield declared local icons; optional/unsupported members stay optional.

    URLs outside the configured origin/project prefix, including data URLs, are
    deliberately outside these read-only local checks. No requests are made.
    """
    manifest = local_reference(public, base, base, manifest_url)
    if manifest is None:
        return
    if not manifest.is_file():
        raise ValueError('Linked manifest missing: ' + manifest_url)
    data = json.loads(manifest.read_text(encoding='utf-8-sig'))
    icons = data.get('icons') if isinstance(data, dict) else None
    if not isinstance(icons, list):
        return
    for icon in icons:
        reference = icon.get('src') if isinstance(icon, dict) else None
        if not isinstance(reference, str) or not _url_input(reference):
            continue
        local = local_reference(public, base, manifest_url, reference)
        if local is None:
            continue
        if not local.is_file():
            raise ValueError('Manifest icon missing: ' + resolve_reference_url(manifest_url, reference))
        yield local

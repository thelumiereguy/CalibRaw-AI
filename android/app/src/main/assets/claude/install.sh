# Run by Termux's bash for CalibRaw (ClaudeBridge) in a visible Termux session, through Termux's
# RUN_COMMAND intent. Installs Claude Code's official linux-arm64 binary and runs it through
# Termux's glibc: the binary is built for glibc, and Termux is built on Android's bionic.
# Safe to run again; it replaces the binary with the release pinned below.
set -euo pipefail
printf '\nCalibRaw is installing Claude Code. This takes a few minutes.\n\n'
termux-wake-lock || true
export DEBIAN_FRONTEND=noninteractive

# Termux installs new packages only onto an up-to-date base.
pkg upgrade -y -o Dpkg::Options::=--force-confold
pkg install -y -o Dpkg::Options::=--force-confold glibc-repo
pkg install -y -o Dpkg::Options::=--force-confold glibc-runner patchelf
loader="$PREFIX/glibc/lib/ld-linux-aarch64.so.1"
[ -x "$loader" ] || { echo "The glibc loader is missing at $loader."; exit 1; }

# The release installed, and its npm registry integrity value (dist.integrity of
# https://registry.npmjs.org/@anthropic-ai/claude-code-linux-arm64/<version>), pinned so the
# install does not change with the registry. Fetched with curl: npm itself hung on a phone.
version=2.1.318
integrity=sha512-FUVVH3aQbLEChjv5wQLdFbpBC7bclVIdVYO5hDzJthDyf2OQtdubIP5QqLPgEScQb3CVIAFg2IPJ5nihJa8Ndw==
tarball="https://registry.npmjs.org/@anthropic-ai/claude-code-linux-arm64/-/claude-code-linux-arm64-$version.tgz"
# The SHA-512, base64 after "sha512-", as hex for sha512sum.
sha512=$(printf '%s' "${integrity#sha512-}" | base64 -d | od -An -v -tx1 | tr -d ' \n')
[ ${#sha512} -eq 128 ] || { echo "The pinned integrity value is not a SHA-512."; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
printf '\nDownloading %s\n' "$tarball"
curl -fL --retry 3 -o "$work/claude.tgz" "$tarball"
printf '%s  %s\n' "$sha512" "$work/claude.tgz" | sha512sum -c -
tar -xzf "$work/claude.tgz" -C "$work"

home="$HOME/.calibraw-claude"
mkdir -p "$home"
install -m 755 "$work/package/claude" "$home/claude.bin"
patchelf --set-interpreter "$loader" "$home/claude.bin"
# termux-exec's LD_PRELOAD is a bionic library, which a glibc program cannot load.
cat > "$PREFIX/bin/claude" <<EOF
#!$PREFIX/bin/sh
unset LD_PRELOAD
export DISABLE_AUTOUPDATER=1
exec "$home/claude.bin" "\$@"
EOF
chmod 755 "$PREFIX/bin/claude"

claude --version
termux-wake-unlock || true
printf '\nClaude Code is installed. Go back to CalibRaw and tap Sign in.\n'

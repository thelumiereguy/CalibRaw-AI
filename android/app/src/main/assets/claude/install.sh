# Run by Termux's bash for CalibRaw (ClaudeBridge) in a visible Termux session, through Termux's
# RUN_COMMAND intent. Installs Claude Code's official linux-arm64 binary and runs it through
# Termux's glibc: the binary is built for glibc, and Termux is built on Android's bionic.
# Safe to run again; it replaces the binary with the newest release.
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

# The registry's metadata, read with curl: npm itself hung on a phone.
metadata=$(curl -fsSL --max-time 60 https://registry.npmjs.org/@anthropic-ai/claude-code-linux-arm64/latest)
tarball=$(printf '%s' "$metadata" | grep -o '"tarball": *"[^"]*"' | cut -d'"' -f4)
# The SHA-512 integrity value, base64 after "sha512-", as hex for sha512sum.
integrity=$(printf '%s' "$metadata" | grep -o '"integrity": *"sha512-[^"]*"' | cut -d'"' -f4)
sha512=$(printf '%s' "${integrity#sha512-}" | base64 -d | od -An -v -tx1 | tr -d ' \n')
[ -n "$tarball" ] && [ ${#sha512} -eq 128 ] || { echo "The registry did not name the package."; exit 1; }

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

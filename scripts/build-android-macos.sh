#!/usr/bin/env bash
# Builds and installs the debug APK on macOS. docs/DEVELOPMENT.md assumes a
# Linux host; on macOS two things differ:
# - Meson (GLib, for Lensfun) needs `ninja` on PATH; the SDK's CMake ships one.
# - GLib's gdbus-codegen imports distutils, which Python 3.12+ removed; a venv
#   with setuptools provides it. CMake's FindPython3 prefers the active venv.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
sdk="${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}"
venv="${CALIBRAW_BUILD_VENV:-$repo_root/../.calibraw-build-venv}"

if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install -q setuptools
fi

# shellcheck disable=SC1091
source "$HOME/.cargo/env"
export ANDROID_SDK_ROOT="$sdk"
export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$sdk/cmake/3.22.1/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 21)}"

cd "$repo_root"
./gradlew assembleDebug --console=plain
"$sdk/platform-tools/adb" install -r android/app/build/outputs/apk/debug/app-debug.apk

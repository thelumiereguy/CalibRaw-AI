# Run by Termux's bash for CalibRaw (ClaudeBridge), through Termux's RUN_COMMAND intent:
#   bash -c "<this script>" calibraw-claude <port> <token> <claude arguments...>
# Connects back to the app's loopback port, proves itself with the app's one-time token, then
# runs claude with its input and output on that connection, in a fresh folder so no project
# settings apply. Once claude exits, one last line tells the app its exit code and the end of
# what it wrote to stderr, base64-encoded so the line stays valid JSON.
port=$1
token=$2
shift 2
exec 3<>"/dev/tcp/127.0.0.1/$port" || exit 90
printf '%s\n' "$token" >&3
work=$(mktemp -d) || exit 91
cd "$work" || exit 91
# Without Claude Code's nonessential traffic: update checks, telemetry, error reporting and
# feature flags, none of which a conversation without tools needs.
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 FORCE_PROMPT_CACHING_5M=1 \
    claude "$@" <&3 >&3 2>"$work/stderr"
code=$?
printf '{"type":"calibraw_exit","code":%d,"stderr":"%s"}\n' \
    "$code" "$(tail -c 2000 "$work/stderr" | base64 -w0)" >&3
cd / && rm -rf "$work"

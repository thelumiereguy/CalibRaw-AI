# Run by Termux's bash for CalibRaw (ClaudeBridge), through Termux's RUN_COMMAND intent, once per
# process of the app before it starts its first conversation:
#   bash -c "<this script>" calibraw-cleanup <generation>
# Ends every claude process that connect.sh started for an earlier process of the app, which it
# marks with that process's generation. When Android kills the app (a reinstall, a force-stop, a
# crash), its end of the connection goes away without closing, so claude would wait for its next
# message without end. Ending claude lets connect.sh finish and remove its folder.
current=$1
for environ in /proc/[0-9]*/environ; do
    # Termux reads only its own processes' environments; the others fail quietly.
    {
        while IFS= read -r -d '' entry; do
            case $entry in
                CALIBRAW_GENERATION=*)
                    if [ "${entry#*=}" != "$current" ]; then
                        pid=${environ#/proc/}
                        kill "${pid%/environ}"
                    fi
                    break
                    ;;
            esac
        done < "$environ"
    } 2>/dev/null
done

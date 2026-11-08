# The Rust side (crates/calibraw-ffi) resolves these classes, methods and
# fields by exact name through JNI, so R8 must neither rename nor remove them.
# `cargo xtask jni-contract` checks the names against the Java sources.

# Native callbacks (`Java_de_duecki_calibraw_CalibRawActivity_*`) plus the
# activity methods and the storageManager/profileImporter/exportPublisher
# fields that Rust reads.
-keep class de.duecki.calibraw.CalibRawActivity { *; }

# Receivers of the JNI calls made through those activity fields.
-keep class de.duecki.calibraw.StorageManager { *; }
-keep class de.duecki.calibraw.ProfileImporter { *; }
-keep class de.duecki.calibraw.ExportPublisher { *; }

-keep class de.duecki.calibraw.ReplayVideoEncoder { *; }
# The AI editor's models: Rust calls AiModels through the activity's aiModels
# field, and its native callbacks are AiModels' own.
-keep class de.duecki.calibraw.ai.AiModels { *; }

# LiteRT-LM's native library calls back into its Kotlin classes, and its AAR
# ships no consumer rules.
-keep class com.google.ai.edge.litertlm.** { *; }

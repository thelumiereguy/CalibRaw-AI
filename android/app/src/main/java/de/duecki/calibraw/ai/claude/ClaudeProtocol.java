package de.duecki.calibraw.ai.claude;

import android.util.Base64;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** What the app writes to {@code claude -p} and reads from it: arguments and stream-json lines. */
final class ClaudeProtocol {
    private ClaudeProtocol() {}

    /**
     * The arguments of a conversation over stream-json on {@code model} with {@code systemPrompt}
     * and nothing else: no tools, settings, MCP servers or saved session, shaped by
     * {@code jsonSchema} when that is not null. Flags without a value are written as
     * {@code --flag=}, so no argument is empty.
     */
    static List<String> conversationArguments(
            String model, String systemPrompt, String jsonSchema) {
        List<String> arguments = new ArrayList<>(Arrays.asList(
                "-p",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", model,
                "--system-prompt", systemPrompt,
                "--tools=",
                "--strict-mcp-config",
                "--mcp-config={\"mcpServers\":{}}",
                "--setting-sources=",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--permission-prompts", "none"));
        if (jsonSchema != null) {
            arguments.add("--json-schema");
            arguments.add(jsonSchema);
        }
        return arguments;
    }

    /** A stream-json user message: the JPEGs at {@code imagePaths}, then {@code text}. */
    static JSONObject userMessage(String text, List<String> imagePaths)
            throws IOException, JSONException {
        JSONArray content = new JSONArray();
        for (String imagePath : imagePaths) {
            JSONObject source = new JSONObject();
            source.put("type", "base64");
            source.put("media_type", "image/jpeg");
            source.put("data", Base64.encodeToString(
                    Files.readAllBytes(Paths.get(imagePath)), Base64.NO_WRAP));
            JSONObject image = new JSONObject();
            image.put("type", "image");
            image.put("source", source);
            content.put(image);
        }
        JSONObject textBlock = new JSONObject();
        textBlock.put("type", "text");
        textBlock.put("text", text);
        content.put(textBlock);
        JSONObject message = new JSONObject();
        message.put("role", "user");
        message.put("content", content);
        JSONObject line = new JSONObject();
        line.put("type", "user");
        line.put("message", message);
        return line;
    }

    /** What a turn cost, from its result line, for the log. */
    static String cost(JSONObject result) {
        return "$" + result.optDouble("total_cost_usd", 0.0);
    }
}

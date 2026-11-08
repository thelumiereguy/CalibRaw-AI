//! The tool calls the model writes as text lines in its reply: the tool's
//! name, then `name=value` pairs, such as `set_sliders shadows=15
//! exposure=0.3`, with a curve's outputs joined by `/`. Claude writes them
//! as assets/claude/editor_system.md describes. Gemma's brief asks for none
//! (brief.rs writes the calls), but Gemma wrote them when a prompt listed the
//! tools, so the chat keeps them out of the text it shows. No tools are
//! declared to the model, so the app reads the calls itself and reads them
//! forgivingly, as Gemma 4 wrote them: pairs without the `=` (`shadows 15`), its own call syntax (`call:set_sliders{shadows:15}`,
//! at times with a brace too many), `:` for `=`, `[…]` lists, quotes and list
//! markers, and lines of pairs without the tool's name (`sharpen_amount=80`),
//! whose values the editor routes to their tools.

use super::super::tools::Call;
use serde_json::{Map, Value};

/// `text` split into what the model said, for the chat, and the calls it wrote.
/// A call starts at a word beginning `set_`, anywhere in a line, and runs to
/// the next such word or the line's end. A line without one that starts with
/// a word `is_value_name` knows is a call with no tool's name.
pub(super) fn read_reply(text: &str, is_value_name: fn(&str) -> bool) -> (String, Vec<Call>) {
    let mut said = Vec::new();
    let mut calls = Vec::new();
    for line in text.lines() {
        let line = line
            .replace("<|tool_call>", " ")
            .replace("<tool_call|>", " ")
            .replace("<|\"|>", "\"");
        let starts = call_starts(&line);
        if starts.is_empty() {
            if let Some(call) = unnamed_call(&line, is_value_name) {
                calls.push(call);
                continue;
            }
        }
        let words = line[..starts.first().copied().unwrap_or(line.len())]
            .trim()
            .trim_end_matches("call:")
            .trim_matches(|c: char| c == '`' || c == '-' || c == '*' || c.is_whitespace());
        if !words.is_empty() {
            said.push(words.to_owned());
        }
        for (index, start) in starts.iter().enumerate() {
            let end = starts.get(index + 1).copied().unwrap_or(line.len());
            match read_call(&line[*start..end]) {
                Some(call) => calls.push(call),
                // A tool named in a sentence, not called.
                None => said.push(line[*start..end].trim().to_owned()),
            }
        }
    }
    (said.join("\n"), calls)
}

/// The byte offsets of the words beginning `set_` in `line`.
fn call_starts(line: &str) -> Vec<usize> {
    line.match_indices("set_")
        .map(|(start, _)| start)
        .filter(|start| {
            line[..*start]
                .chars()
                .next_back()
                .is_none_or(|before| !before.is_alphanumeric() && before != '_')
        })
        .collect()
}

/// `line` as a call with no tool's name, when its first word is a value
/// name: Gemma wrote `sharpen_amount=80`, which set nothing, despite the
/// prompt saying each line starts with its tool's name.
fn unnamed_call(line: &str, is_value_name: fn(&str) -> bool) -> Option<Call> {
    let line =
        line.trim_start_matches(|c: char| c == '`' || c == '-' || c == '*' || c.is_whitespace());
    let first_end = line
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(line.len());
    if !is_value_name(&line[..first_end].to_lowercase()) {
        return None;
    }
    // The leading space leaves the call's name empty.
    read_call(&format!(" {line}"))
}

/// One call, `name` then its pairs; `None` without a pair.
fn read_call(text: &str) -> Option<Call> {
    let name_end = text
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(text.len());
    let name = text[..name_end].to_lowercase();
    let mut rest = lists_joined(&text[name_end..]);
    for separator in ['{', '}', '(', ')', ',', ';', '"', '\'', '`'] {
        rest = rest.replace(separator, " ");
    }
    let rest = rest.replace(['=', ':'], " = ");
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let mut arguments = Map::new();
    let mut index = 0;
    while index + 1 < tokens.len() {
        let (name, next) = (tokens[index], tokens[index + 1]);
        if next == "=" {
            if let Some(written) = tokens.get(index + 2).filter(|written| **written != "=") {
                arguments.insert(name.to_lowercase(), value(written));
                index += 3;
                continue;
            }
        } else if looks_like_value(next) && !looks_like_value(name) {
            // `shadows 15`: without an example of the syntax in the prompt,
            // Gemma wrote most pairs without the `=`.
            arguments.insert(name.to_lowercase(), value(next));
            index += 2;
            continue;
        }
        index += 1;
    }
    (!arguments.is_empty()).then_some(Call {
        name,
        arguments: Value::Object(arguments),
    })
}

/// `text` with each `[a, b, c]` list written `a/b/c`, the way a curve's
/// outputs are.
fn lists_joined(text: &str) -> String {
    let mut joined = String::with_capacity(text.len());
    let mut in_list = false;
    for c in text.chars() {
        match c {
            '[' => in_list = true,
            ']' => in_list = false,
            ',' if in_list => joined.push('/'),
            c if in_list && c.is_whitespace() => {}
            c => joined.push(c),
        }
    }
    joined
}

/// Whether `text` is a value rather than a name: `true`, `false`, or text
/// with a digit, such as `15`, `-0.5`, `0→20` or `0/64/128/192/255`; no
/// value name has a digit.
fn looks_like_value(text: &str) -> bool {
    matches!(text.to_lowercase().as_str(), "true" | "false")
        || text.chars().any(|c| c.is_ascii_digit())
}

/// A value as the model wrote it: `true`/`false`, a `/`-joined list, a number,
/// or else the text, which the tool then reports as not set. A `was→now`
/// pair, copied from the app's own messages, is its new value.
fn value(text: &str) -> Value {
    let text = text.rsplit('→').next().unwrap_or(text);
    let text = text.trim_end_matches(['.', '%']);
    match text.to_lowercase().as_str() {
        "true" | "on" => return Value::Bool(true),
        "false" | "off" => return Value::Bool(false),
        _ => {}
    }
    if text.contains('/') {
        return Value::Array(text.split('/').map(value).collect());
    }
    text.parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or_else(|| Value::String(text.to_owned()), Value::Number)
}

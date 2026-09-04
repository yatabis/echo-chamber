use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::{Map, Value};

use super::chat::EchoToolContract;

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// One E.C.H.O. `ModelOutputItem` produced from Qwen's native text envelope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EchoOutputItem {
    /// Natural-language assistant output.
    Message {
        /// Fixed provider-neutral assistant role.
        role: EchoAssistantRole,
        /// Exact model text outside recognized tool-call envelopes.
        content: String,
    },
    /// Parsed Qwen function call.
    ToolCall {
        /// Adapter-stable call identity.
        call_id: String,
        /// Parsed function name.
        tool_name: String,
        /// Compact JSON object containing parsed parameters.
        input: String,
    },
}

/// Exact assistant discriminator serialized on message output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EchoAssistantRole {
    /// Model assistant.
    Assistant,
}

/// Structured output plus a non-fatal parser warning.
///
/// Generation has already advanced state when parsing runs. Malformed
/// tool-like output therefore remains an assistant message rather than turning
/// an already committed state transition into an apparent failed transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedQwenOutput {
    /// Provider-neutral output items.
    pub output: Vec<EchoOutputItem>,
    /// Explanation when tool-like text could not be safely structured.
    pub warning: Option<String>,
}

/// Parses Qwen's admitted XML-like function-call envelope.
#[must_use]
pub fn parse_qwen_output(text: &str, request_id: &str) -> ParsedQwenOutput {
    parse_qwen_output_with_tools(text, request_id, &[])
}

/// Parses Qwen tool output using the request's exact JSON Schemas.
///
/// XML parameter bodies carry no intrinsic type marker. Schema-less values
/// therefore remain strings instead of guessing from JSON-looking text.
#[must_use]
pub fn parse_qwen_output_with_tools(
    text: &str,
    request_id: &str,
    tools: &[EchoToolContract],
) -> ParsedQwenOutput {
    let Some(first_tool) = text.find(TOOL_CALL_OPEN) else {
        return ParsedQwenOutput {
            output: message_output(text),
            warning: None,
        };
    };

    match parse_tool_sequence(text, first_tool, request_id, tools) {
        Ok(output) => ParsedQwenOutput {
            output,
            warning: None,
        },
        Err(detail) => ParsedQwenOutput {
            output: message_output(text),
            warning: Some(detail),
        },
    }
}

fn parse_tool_sequence(
    text: &str,
    first_tool: usize,
    request_id: &str,
    tools: &[EchoToolContract],
) -> Result<Vec<EchoOutputItem>, String> {
    let mut output = message_output(text[..first_tool].trim_end());
    let mut remaining = &text[first_tool..];
    let mut ordinal = 1usize;

    loop {
        let trimmed = remaining.trim_start();
        if !trimmed.starts_with(TOOL_CALL_OPEN) {
            return Err("text after a tool call is not another complete tool_call envelope".into());
        }
        let after_open = &trimmed[TOOL_CALL_OPEN.len()..];
        let close = after_open
            .find(TOOL_CALL_CLOSE)
            .ok_or_else(|| "tool_call envelope is missing </tool_call>".to_string())?;
        let (tool_name, input) = parse_function(&after_open[..close], tools)?;
        output.push(EchoOutputItem::ToolCall {
            call_id: format!("{request_id}:tool:{ordinal}"),
            tool_name,
            input,
        });
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| "tool-call ordinal overflow".to_string())?;
        remaining = &after_open[close + TOOL_CALL_CLOSE.len()..];
        if remaining.trim().is_empty() {
            return Ok(output);
        }
    }
}

fn parse_function(block: &str, tools: &[EchoToolContract]) -> Result<(String, String), String> {
    let block = block.trim();
    let after_open = block
        .strip_prefix(FUNCTION_OPEN)
        .ok_or_else(|| "tool_call must contain one <function=name> block".to_string())?;
    let name_end = after_open
        .find('>')
        .ok_or_else(|| "function opening tag is missing '>'".to_string())?;
    let name = &after_open[..name_end];
    validate_name(name, "function")?;
    let after_name = &after_open[name_end + 1..];
    let function_close = after_name
        .rfind(FUNCTION_CLOSE)
        .ok_or_else(|| "function block is missing </function>".to_string())?;
    if !after_name[function_close + FUNCTION_CLOSE.len()..]
        .trim()
        .is_empty()
    {
        return Err("tool_call contains text after </function>".into());
    }

    let input_schema = tools
        .iter()
        .find(|tool| tool.name == name)
        .map(|tool| &tool.input_schema);
    let parameters = parse_parameters(&after_name[..function_close], input_schema)?;
    let input = serde_json::to_string(&Value::Object(parameters))
        .map_err(|error| format!("could not serialize parsed tool input: {error}"))?;
    Ok((name.into(), input))
}

fn parse_parameters(
    mut input: &str,
    input_schema: Option<&Value>,
) -> Result<Map<String, Value>, String> {
    let mut parameters = Map::new();
    let mut names = BTreeSet::new();
    loop {
        input = input.trim_start();
        if input.is_empty() {
            return Ok(parameters);
        }
        let after_open = input
            .strip_prefix(PARAMETER_OPEN)
            .ok_or_else(|| "function body contains text outside a parameter block".to_string())?;
        let name_end = after_open
            .find('>')
            .ok_or_else(|| "parameter opening tag is missing '>'".to_string())?;
        let name = &after_open[..name_end];
        validate_name(name, "parameter")?;
        if !names.insert(name.to_string()) {
            return Err(format!("duplicate parameter {name:?}"));
        }
        let after_name = &after_open[name_end + 1..];
        let value_end = after_name
            .find(PARAMETER_CLOSE)
            .ok_or_else(|| format!("parameter {name:?} is missing </parameter>"))?;
        let raw_value = after_name[..value_end].trim();
        let value = parse_parameter_value(raw_value, parameter_schema(input_schema, name))
            .map_err(|detail| format!("parameter {name:?} {detail}"))?;
        parameters.insert(name.into(), value);
        input = &after_name[value_end + PARAMETER_CLOSE.len()..];
    }
}

fn parameter_schema<'a>(input_schema: Option<&'a Value>, name: &str) -> Option<&'a Value> {
    let schema = input_schema?.as_object()?;
    if let Some(property) = schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(name))
    {
        return Some(property);
    }
    schema
        .get("additionalProperties")
        .filter(|additional| additional.is_object())
}

fn parse_parameter_value(raw_value: &str, schema: Option<&Value>) -> Result<Value, String> {
    let Some(schema) = schema else {
        return Ok(Value::String(raw_value.into()));
    };
    let mut declared_types = BTreeSet::new();
    collect_declared_types(schema, &mut declared_types);
    if declared_types.is_empty() {
        return Ok(Value::String(raw_value.into()));
    }
    if let Ok(parsed) = serde_json::from_str::<Value>(raw_value)
        && declared_types.contains(json_type(&parsed))
    {
        return Ok(parsed);
    }
    if declared_types.contains("string") {
        return Ok(Value::String(raw_value.into()));
    }
    Err(format!(
        "does not match declared JSON type(s): {}",
        declared_types.into_iter().collect::<Vec<_>>().join(", ")
    ))
}

fn collect_declared_types(schema: &Value, output: &mut BTreeSet<&'static str>) {
    if let Some(declared) = schema.get("type") {
        match declared {
            Value::String(kind) => insert_declared_type(kind, output),
            Value::Array(kinds) => {
                for kind in kinds.iter().filter_map(Value::as_str) {
                    insert_declared_type(kind, output);
                }
            }
            _ => {}
        }
    }
    for alternatives in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = schema.get(alternatives).and_then(Value::as_array) {
            for branch in branches {
                collect_declared_types(branch, output);
            }
        }
    }
    if let Some(constant) = schema.get("const") {
        output.insert(json_type(constant));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        output.extend(values.iter().map(json_type));
    }
}

fn insert_declared_type(kind: &str, output: &mut BTreeSet<&'static str>) {
    match kind {
        "integer" | "number" => {
            output.insert("number");
        }
        "array" => {
            output.insert("array");
        }
        "boolean" => {
            output.insert("boolean");
        }
        "null" => {
            output.insert("null");
        }
        "object" => {
            output.insert("object");
        }
        "string" => {
            output.insert("string");
        }
        _ => {}
    }
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn validate_name(value: &str, label: &str) -> Result<(), String> {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Ok(());
    }
    Err(format!("{label} name {value:?} is not protocol-safe"))
}

fn message_output(text: &str) -> Vec<EchoOutputItem> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![EchoOutputItem::Message {
            role: EchoAssistantRole::Assistant,
            content: text.into(),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::EchoToolContract;
    use serde_json::json;

    #[test]
    fn plain_text_remains_one_message() {
        assert_eq!(
            parse_qwen_output("考えています。", "rin:1"),
            ParsedQwenOutput {
                output: vec![EchoOutputItem::Message {
                    role: EchoAssistantRole::Assistant,
                    content: "考えています。".into(),
                }],
                warning: None,
            }
        );
    }

    #[test]
    fn schema_restores_numbers_booleans_and_numeric_looking_strings() {
        let tools = [tool_contract(
            "check_notifications",
            &json!({
                "limit": { "type": "integer" },
                "channel": { "type": "string" },
                "silent": { "type": "boolean" }
            }),
        )];
        let parsed = parse_qwen_output_with_tools(
            "<tool_call>\n\
             <function=check_notifications>\n\
             <parameter=limit>\n20\n</parameter>\n\
             <parameter=channel>\n123456789\n</parameter>\n\
             <parameter=silent>\ntrue\n</parameter>\n\
             </function>\n\
             </tool_call>",
            "rin:7",
            &tools,
        );
        assert_eq!(
            parsed,
            ParsedQwenOutput {
                output: vec![EchoOutputItem::ToolCall {
                    call_id: "rin:7:tool:1".into(),
                    tool_name: "check_notifications".into(),
                    input: r#"{"limit":20,"channel":"123456789","silent":true}"#.into(),
                }],
                warning: None,
            }
        );
    }

    #[test]
    fn missing_schema_never_guesses_a_parameter_type() {
        let parsed = parse_qwen_output(
            "<tool_call><function=unknown><parameter=value>123456789</parameter>\
             </function></tool_call>",
            "rin:8",
        );
        assert!(matches!(
            &parsed.output[0],
            EchoOutputItem::ToolCall { input, .. } if input == r#"{"value":"123456789"}"#
        ));
    }

    #[test]
    fn reasoning_prefix_and_multiple_calls_keep_order() {
        let tools = [tool_contract(
            "second",
            &json!({ "value": { "type": "boolean" } }),
        )];
        let parsed = parse_qwen_output_with_tools(
            "先に確認します。\n\n\
             <tool_call><function=first></function></tool_call>\n\
             <tool_call><function=second><parameter=value>true</parameter>\
             </function></tool_call>",
            "marie:2",
            &tools,
        );
        assert_eq!(parsed.output.len(), 3);
        assert!(matches!(
            &parsed.output[0],
            EchoOutputItem::Message { content, .. } if content == "先に確認します。"
        ));
        assert!(matches!(
            &parsed.output[2],
            EchoOutputItem::ToolCall { tool_name, input, .. }
                if tool_name == "second" && input == r#"{"value":true}"#
        ));
        assert!(parsed.warning.is_none());
    }

    #[test]
    fn malformed_tool_text_remains_visible_with_warning() {
        let text = "<tool_call><function=unsafe/name></function></tool_call>";
        let parsed = parse_qwen_output(text, "rin:3");
        assert_eq!(parsed.output, message_output(text));
        assert!(parsed.warning.is_some());
    }

    fn tool_contract(name: &str, properties: &Value) -> EchoToolContract {
        EchoToolContract {
            name: name.into(),
            description: "test tool".into(),
            input_schema: json!({
                "type": "object",
                "properties": properties,
                "additionalProperties": false
            }),
            output_schema: None,
            strict: true,
        }
    }
}

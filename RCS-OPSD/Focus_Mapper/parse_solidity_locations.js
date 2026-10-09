#!/usr/bin/env node

const parser = require("@solidity-parser/parser");

function typeKey(node) {
  if (!node) return "?";
  switch (node.type) {
    case "ElementaryTypeName":
      return node.name + (node.stateMutability === "payable" ? " payable" : "");
    case "UserDefinedTypeName":
      return node.namePath;
    case "ArrayTypeName":
      return `${typeKey(node.baseTypeName)}[${expressionKey(node.length)}]`;
    case "Mapping":
      return `mapping(${typeKey(node.keyType)}=>${typeKey(node.valueType)})`;
    case "FunctionTypeName": {
      const parameters = (node.parameterTypes || []).map(typeKey).join(",");
      const returns = (node.returnTypes || []).map(typeKey).join(",");
      return `function(${parameters})returns(${returns})`;
    }
    default:
      return node.type || "?";
  }
}

function expressionKey(node) {
  if (!node) return "";
  if (node.type === "NumberLiteral") return node.number;
  if (node.type === "Identifier") return node.name;
  if (node.type === "MemberAccess") {
    return `${expressionKey(node.expression)}.${node.memberName}`;
  }
  if (node.type === "BinaryOperation") {
    return `${expressionKey(node.left)}${node.operator}${expressionKey(node.right)}`;
  }
  return node.type;
}

function functionName(node) {
  if (node.isConstructor) return "constructor";
  if (node.isReceiveEther) return "receive";
  if (node.isFallback) return "fallback";
  return node.name || "fallback";
}

function functionDescriptor(node, contract) {
  const name = functionName(node);
  const parameters = (node.parameters || []).map((parameter) => typeKey(parameter.typeName));
  return {
    kind: "function",
    contract,
    name,
    symbol_key: `${contract}::${name}(${parameters.join(",")})`,
    start_line: node.loc.start.line,
    end_line: node.loc.end.line,
  };
}

function stateDescriptor(variable, contract) {
  return {
    kind: "state_variable_getter",
    contract,
    name: variable.name,
    symbol_key: `${contract}::state::${variable.name}:${typeKey(variable.typeName)}`,
    start_line: variable.loc.start.line,
    end_line: variable.loc.end.line,
  };
}

function extract(source) {
  const ast = parser.parse(source, { loc: true });
  const output = [];
  for (const child of ast.children || []) {
    if (child.type === "FunctionDefinition") {
      output.push(functionDescriptor(child, "<top-level>"));
      continue;
    }
    if (child.type !== "ContractDefinition") continue;
    for (const member of child.subNodes || []) {
      if (member.type === "FunctionDefinition") {
        output.push(functionDescriptor(member, child.name));
      } else if (member.type === "StateVariableDeclaration") {
        for (const variable of member.variables || []) {
          if (variable.visibility === "public") {
            output.push(stateDescriptor(variable, child.name));
          }
        }
      }
    }
  }
  return output;
}

let source = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  source += chunk;
});
process.stdin.on("end", () => {
  try {
    process.stdout.write(JSON.stringify(extract(source)));
  } catch (error) {
    process.stderr.write(String(error && error.stack ? error.stack : error));
    process.exitCode = 1;
  }
});


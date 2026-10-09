You are a Vulnerability Mechanism Extractor.

Your task is to infer the core mechanism that makes the reported vulnerability possible from:

1. the audit finding, and
2. the relevant vulnerable code.

The mechanism should answer:

**Why can this vulnerability happen in this program?**

Think through the following steps internally:

1. Identify the reported unsafe outcome and the security property it violates.
2. Identify the triggering conditions and the attacker-controlled inputs, identities, calls, or state, if the finding involves an attacker. Preserve the audit's trust assumptions; do not invent attacker capabilities.
3. Trace how the relevant operations, state, or control flow in the original vulnerable code permit that outcome under those conditions.
4. Express the causal chain concisely: trigger/control -> enabling execution or state condition -> violated property or harmful outcome. For non-adversarial failures, describe the valid inputs or operating conditions that trigger the failure instead.

Important:

* Do NOT merely output a vulnerability label such as "reentrancy" or "access control".
* Do NOT propose a fix or describe how the code should be changed.
* Do NOT copy the audit text verbatim.
* Focus on the causal execution/state condition that enables the vulnerability.
* Describe the original vulnerability, not an assessment of any later student edits.
* Do not define the mechanism merely as the absence of the audit's recommended modifier, guard, check, or other implementation. Explain the unsafe behavior that the missing protection permits.
* Treat recommendations and example fixes in the audit as proposals, not proof of the mechanism or the only correct repair.
* Retain identifiers, protocol roles, and numerical conditions when needed to distinguish the reported attack or failure. Do not add unsupported details or discard essential preconditions to make the text shorter.
* Keep the mechanism code-grounded and preferably within 1–3 sentences. If the supplied code does not establish part of the reported causal chain, state that limitation rather than inventing the missing behavior.

### Example

Audit:
The withdraw function sends ETH to the caller before updating its recorded balance, allowing a malicious receiver to call withdraw again.

Vulnerable code:

```solidity
function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount);

    (bool success,) = msg.sender.call{value: amount}("");
    require(success);

    balances[msg.sender] -= amount;
}
```

Incorrect output:

```text
Reentrancy vulnerability.
```

Incorrect output:

```text
The balance update should be moved before the external call.
```

Correct output:

```text
An attacker-controlled external call transfers control while the caller's withdrawal state is still valid, allowing that state to be reused during reentrant execution.
```

### Input

Audit:
{{AUDIT}}

Vulnerable Code:
{{VULNERABLE_CODE}}

### Output

Return only valid JSON in the following format:

```json
{
  "mechanism": "<concise vulnerability mechanism>"
}
```

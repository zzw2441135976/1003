You are a Repair Invariant Extractor.

Your task is to infer the **patch-independent security condition that a correct repair must satisfy**.

The repair invariant should answer:

**What security condition must hold after repair so that the vulnerability mechanism can no longer occur?**

Think through the following steps internally:

1. Derive the reported unsafe outcome, triggering conditions, and trust assumptions from the audit, mechanism, causal site, and original vulnerable code. Do this before interpreting the reference patch.
2. Determine the semantic security condition needed to rule out that outcome under the reported threat model while preserving intended legitimate behavior.
3. Compare the reference before/after excerpts and diff to identify which part of that condition the patch attempts to enforce. The reference is evidence of an intended repair, not proof that the vulnerability is fully repaired.
4. Check a counterexample: could the original attack or failure still occur while the proposed invariant holds? If so, refine the condition. Include audit-supported bypasses, such as attacker-controlled alternate identities or contracts, when relevant.
5. Check alternative repairs: could a correct repair satisfy the required security property through different storage, control flow, authorization, or enforcement but violate this wording? If so, remove the implementation-specific requirement.
6. Output the resulting security condition, not a summary of the reference change. If evidence is insufficient to establish a complete condition, explicitly qualify the unsupported part instead of treating the reference patch as a complete repair.

Important rules:

* Describe **what must hold**, not **how to implement the repair**.
* Do NOT copy the reference patch or prescribe its specific code changes.
* Do NOT output instructions such as "add a modifier", "move this statement", or "use a reentrancy guard".
* Do NOT make the invariant depend on the reference patch's specific implementation, including its particular variable names, newly introduced functions or modifiers, storage representation, control-flow structure, or enforcement technique.
* Prefer semantic roles over concrete identifiers when an identifier reflects only the reference repair's implementation. Concrete identifiers may be retained when necessary to identify an affected protocol object or causal site, but the invariant must not require a correct repair to use that identifier or representation.
* The invariant must permit alternative correct implementations that enforce the same security condition.
* The invariant should directly prevent a necessary condition of the given vulnerability mechanism.
* It must block the reported unsafe outcome, including the bypasses supported by the audit; blocking only the reference patch's chosen example is insufficient.
* Do not turn a reference-only constant, iteration threshold, statement ordering, identity comparison, or representation into a mandatory invariant unless the audit and original code establish it as a semantic requirement. For example, an address inequality does not establish independent beneficiaries when an attacker controls multiple addresses; a contract-code check does not establish trustworthy behavior.
* Preserve intended legitimate operations. Disabling all affected functionality is not an adequate security invariant unless that is explicitly the task's intended behavior.
* Keep the invariant specific enough to capture the security requirement, but general enough to allow different correct repair implementations.
* Do NOT merely state that "the vulnerability must be prevented."

### Example

Vulnerability Mechanism:

An attacker-controlled external call transfers control while the caller's withdrawal state remains valid, allowing that state to be reused during reentrant execution.

Causal Site:

```json
{
  "functions": [
    "src/Vault.sol::Vault::withdraw(uint256)"
  ],
  "state_variables": [
    "Vault.balances"
  ],
  "operations": [
    "external call to msg.sender",
    "withdrawal state remains valid during the external call"
  ]
}
```

Vulnerable Code:

```solidity
function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount);

    (bool success,) = msg.sender.call{value: amount}("");
    require(success);

    balances[msg.sender] -= amount;
}
```

Fixed Code:

```solidity
function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount);

    balances[msg.sender] -= amount;

    (bool success,) = msg.sender.call{value: amount}("");
    require(success);
}
```

Incorrect:

```text
Move the balance update before the external call.
```

This describes the specific reference repair rather than the security condition it enforces.

Incorrect:

```text
The contract must prevent reentrancy.
```

This is too general and does not capture the causal condition that must be eliminated.

Correct:

```text
Attacker-controlled reentrant execution must not be able to reuse a still-valid withdrawal state.
```

### Input

Audit Finding and Reported Threat Model:

{{AUDIT}}

Vulnerability Mechanism:

{{MECHANISM}}

Causal Site:

{{CAUSAL_SITE}}

Original Vulnerable Code Context:

{{VULNERABLE_CONTEXT}}

Reference Before-Patch Excerpts (patch-local, not necessarily complete functions):

{{VULNERABLE_CODE}}

Reference After-Patch Excerpts (candidate repair, not assumed correct):

{{FIXED_CODE}}

Reference Patch / Diff:

{{PATCH_DIFF}}

### Output

Return only valid JSON in the following format:

```json
{
  "invariant": "<concise patch-independent security invariant>"
}
```

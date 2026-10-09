import { appendFileSync } from "node:fs";

// Observe the final provider payload, after compaction and message conversion.
// Returning nothing leaves the student's request unchanged.
export default function (pi) {
  let turnId = -1;
  pi.on("turn_start", () => { turnId += 1; });
  pi.on("before_provider_request", (event) => {
    appendFileSync("/config/provider-requests.jsonl",
      JSON.stringify({ turn_id: turnId, request: event.payload }) + "\n");
  });
}

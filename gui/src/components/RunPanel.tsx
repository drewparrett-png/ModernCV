import { useStore } from "../store";

export function RunPanel() {
  const run = useStore((s) => s.run);
  const running = useStore((s) => s.running);
  const result = useStore((s) => s.runResult);
  const error = useStore((s) => s.runError);

  return (
    <div className="run-panel">
      <button onClick={run} disabled={running}>
        {running ? "Running…" : "Run"}
      </button>
      {error && <pre className="run-error">{error}</pre>}
      {result && <pre className="run-result">{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

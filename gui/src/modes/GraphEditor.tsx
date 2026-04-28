/**
 * Graph Editor mode — the original block-graph view, kept as an "advanced"
 * tab. Useful for inspecting what the Learn wizard is assembling under the
 * hood and for one-off experimental graphs.
 */

import { Canvas } from "../components/Canvas";
import { Palette } from "../components/Palette";
import { RunPanel } from "../components/RunPanel";

export function GraphEditor() {
  return (
    <div className="graph-mode">
      <Palette />
      <main className="canvas-pane">
        <Canvas />
      </main>
      <RunPanel />
    </div>
  );
}

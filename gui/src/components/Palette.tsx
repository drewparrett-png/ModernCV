import { useStore } from "../store";
import { BLOCK_KINDS } from "../types";
import { blockMeta, implMeta } from "../blockMeta";

export function Palette() {
  const blocks = useStore((s) => s.blocks);
  return (
    <aside className="palette">
      <h2>Blocks</h2>
      <ul>
        {BLOCK_KINDS.map((kind) => {
          const meta = blockMeta(kind);
          const impls = blocks[kind];
          const implLabels = impls
            ? impls.map((id) => implMeta(kind, id).label)
            : null;
          return (
            <li key={kind}>
              <div className="palette-kind" title={meta.description}>
                {meta.label}
              </div>
              <div className="palette-desc">{meta.description}</div>
              <div className="palette-impls">
                {implLabels ? implLabels.join(", ") : "…"}
              </div>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}

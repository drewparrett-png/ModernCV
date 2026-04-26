import { Handle, Position, type NodeProps } from "reactflow";
import { useStore } from "../store";
import type { BlockNodeData } from "../types";

export function BlockNode({ id, data }: NodeProps<BlockNodeData>) {
  const blocks = useStore((s) => s.blocks);
  const setImpl = useStore((s) => s.setImpl);
  const setParam = useStore((s) => s.setParam);

  const impls = blocks[data.kind] ?? [];
  const showPath = data.kind === "input" || data.kind === "output";
  const path = (data.params.path as string | undefined) ?? "";

  return (
    <div className="block-node">
      <Handle type="target" position={Position.Left} />
      <div className="block-kind">{data.kind}</div>
      <select
        value={data.impl}
        onChange={(e) => setImpl(id, e.target.value)}
        className="nodrag"
      >
        {/* If the current impl isn't in the loaded list, show it anyway. */}
        {!impls.includes(data.impl) && (
          <option value={data.impl}>{data.impl}</option>
        )}
        {impls.map((impl) => (
          <option key={impl} value={impl}>
            {impl}
          </option>
        ))}
      </select>
      {showPath && (
        <input
          type="text"
          value={path}
          placeholder="path"
          onChange={(e) => setParam(id, "path", e.target.value)}
          className="nodrag"
        />
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

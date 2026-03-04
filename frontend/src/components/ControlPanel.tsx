import type { FilterOptions, GraphStats } from '../types';

interface ControlPanelProps {
  filters: FilterOptions;
  stats: GraphStats;
  communities: number[];
  onFiltersChange: (filters: FilterOptions) => void;
}

export function ControlPanel({
  filters,
  stats,
  communities,
  onFiltersChange,
}: ControlPanelProps) {
  return (
    <div className="absolute top-4 left-4 w-72 bg-slate-800/95 backdrop-blur rounded-lg shadow-xl border border-slate-700">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <h1 className="text-lg font-bold text-white">SemLink</h1>
        <p className="text-xs text-slate-400 mt-1">Knowledge Graph Explorer</p>
      </div>

      {/* Stats */}
      <div className="p-4 border-b border-slate-700 grid grid-cols-2 gap-3">
        <div className="text-center">
          <div className="text-2xl font-bold text-indigo-400">{stats.nodeCount}</div>
          <div className="text-xs text-slate-400">Nodes</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-emerald-400">{stats.edgeCount}</div>
          <div className="text-xs text-slate-400">Edges</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-amber-400">{stats.communities}</div>
          <div className="text-xs text-slate-400">Communities</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-rose-400">{(stats.avgWeight * 100).toFixed(0)}%</div>
          <div className="text-xs text-slate-400">Avg Weight</div>
        </div>
      </div>

      {/* Filters */}
      <div className="p-4 space-y-4">
        {/* Search */}
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">
            Search
          </label>
          <input
            type="text"
            value={filters.searchQuery}
            onChange={(e) =>
              onFiltersChange({ ...filters, searchQuery: e.target.value })
            }
            placeholder="Search notes..."
            className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          />
        </div>

        {/* Min Weight */}
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">
            Min Weight: {(filters.minWeight * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={filters.minWeight}
            onChange={(e) =>
              onFiltersChange({ ...filters, minWeight: parseFloat(e.target.value) })
            }
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
        </div>

        {/* Community Filter */}
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">
            Community
          </label>
          <select
            value={filters.community ?? ''}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                community: e.target.value === '' ? null : parseInt(e.target.value),
              })
            }
            className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:outline-none focus:border-indigo-500"
          >
            <option value="">All Communities</option>
            {communities.map((c) => (
              <option key={c} value={c}>
                Community {c}
              </option>
            ))}
          </select>
        </div>

        {/* Reset */}
        <button
          onClick={() =>
            onFiltersChange({ minWeight: 0, community: null, searchQuery: '' })
          }
          className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded transition-colors"
        >
          Reset Filters
        </button>
      </div>

      {/* Help */}
      <div className="p-4 border-t border-slate-700">
        <p className="text-xs text-slate-500">
          Drag nodes to rearrange. Scroll to zoom. Click node for details.
        </p>
      </div>
    </div>
  );
}

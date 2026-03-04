import type { FilterOptions, GraphStats, Topic } from '../types';

// Same colors as ForceGraph/TopicPanel for consistency
const TOPIC_COLORS = [
  '#6366f1', // indigo
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f59e0b', // amber
  '#8b5cf6', // violet
  '#10b981', // emerald
  '#f43f5e', // rose
  '#06b6d4', // cyan
  '#84cc16', // lime
  '#a855f7', // purple
];

interface ControlPanelProps {
  filters: FilterOptions;
  stats: GraphStats;
  communities: number[];
  topics: Topic[];
  onFiltersChange: (filters: FilterOptions) => void;
}

export function ControlPanel({
  filters,
  stats,
  communities,
  topics,
  onFiltersChange,
}: ControlPanelProps) {
  // Create a map from community/topic ID to label
  const topicLabels: Record<number, string> = {};
  topics.forEach((t) => {
    topicLabels[t.id] = t.label;
  });

  const getTopicColor = (topicId: number) => {
    return TOPIC_COLORS[topicId % TOPIC_COLORS.length];
  };

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
          <div className="text-2xl font-bold text-amber-400">{topics.length || stats.communities}</div>
          <div className="text-xs text-slate-400">Topics</div>
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

        {/* Topic Filter */}
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">
            Topic
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
            <option value="">All Topics</option>
            {communities.map((c) => (
              <option key={c} value={c}>
                {topicLabels[c] || `Topic ${c}`}
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

      {/* Topic Legend */}
      {topics.length > 0 && (
        <div className="p-4 border-t border-slate-700">
          <p className="text-xs font-medium text-slate-300 mb-2">Topic Colors</p>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {topics.slice(0, 8).map((topic) => (
              <div key={topic.id} className="flex items-center gap-2 text-xs">
                <div
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: getTopicColor(topic.id) }}
                />
                <span className="text-slate-400 truncate">{topic.label}</span>
              </div>
            ))}
            {topics.length > 8 && (
              <p className="text-xs text-slate-500 pl-4">+{topics.length - 8} more</p>
            )}
          </div>
        </div>
      )}

      {/* Help */}
      <div className="p-4 border-t border-slate-700">
        <p className="text-xs text-slate-500">
          Drag nodes to rearrange. Scroll to zoom. Click node for details.
        </p>
      </div>
    </div>
  );
}

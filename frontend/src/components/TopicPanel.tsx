import { useState } from 'react';
import type { Topic } from '../types';

// Same colors as ForceGraph for consistency
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

interface TopicPanelProps {
  topics: Topic[];
  orphanNotes: string[];
  selectedTopic: number | null;
  onTopicSelect: (topicId: number | null) => void;
  loading?: boolean;
}

export function TopicPanel({
  topics,
  orphanNotes,
  selectedTopic,
  onTopicSelect,
  loading = false,
}: TopicPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [expandedTopic, setExpandedTopic] = useState<number | null>(null);

  const getTopicColor = (topicId: number) => {
    return TOPIC_COLORS[topicId % TOPIC_COLORS.length];
  };

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="absolute top-4 right-4 p-3 bg-slate-800/95 backdrop-blur rounded-lg shadow-xl border border-slate-700 hover:bg-slate-700 transition-colors"
        title="Show Topics"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5 text-slate-300"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
          />
        </svg>
      </button>
    );
  }

  return (
    <div className="absolute top-4 right-4 w-80 max-h-[calc(100vh-2rem)] bg-slate-800/95 backdrop-blur rounded-lg shadow-xl border border-slate-700 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-slate-700 flex items-center justify-between shrink-0">
        <div>
          <h2 className="text-lg font-bold text-white">Topics</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            {topics.length} topic{topics.length !== 1 ? 's' : ''} detected
          </p>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          className="p-1.5 hover:bg-slate-700 rounded transition-colors"
          title="Collapse"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 text-slate-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="p-4 text-center text-slate-400">
          <div className="animate-spin inline-block w-5 h-5 border-2 border-slate-600 border-t-indigo-500 rounded-full mb-2"></div>
          <p className="text-sm">Detecting topics...</p>
        </div>
      )}

      {/* Topics list */}
      {!loading && topics.length > 0 && (
        <div className="overflow-y-auto flex-1 p-2">
          {/* Clear selection button */}
          {selectedTopic !== null && (
            <button
              onClick={() => onTopicSelect(null)}
              className="w-full mb-2 px-3 py-2 bg-slate-700/50 hover:bg-slate-700 text-slate-300 text-sm rounded transition-colors flex items-center gap-2"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Show all topics
            </button>
          )}

          {/* Topic cards */}
          {topics.map((topic) => {
            const isSelected = selectedTopic === topic.id;
            const isExpanded = expandedTopic === topic.id;
            const color = getTopicColor(topic.id);

            return (
              <div
                key={topic.id}
                className={`mb-2 rounded-lg border transition-all ${
                  isSelected
                    ? 'border-indigo-500 bg-slate-700/80'
                    : 'border-slate-600/50 bg-slate-700/30 hover:bg-slate-700/50'
                }`}
              >
                {/* Topic header - clickable to filter */}
                <button
                  onClick={() => onTopicSelect(isSelected ? null : topic.id)}
                  className="w-full p-3 text-left"
                >
                  <div className="flex items-start gap-3">
                    {/* Color indicator */}
                    <div
                      className="w-3 h-3 rounded-full mt-1 shrink-0"
                      style={{ backgroundColor: color }}
                    />

                    <div className="flex-1 min-w-0">
                      {/* Label */}
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="font-medium text-white truncate">
                          {topic.label}
                        </h3>
                        <span className="text-xs bg-slate-600 text-slate-300 px-1.5 py-0.5 rounded shrink-0">
                          {topic.size} note{topic.size !== 1 ? 's' : ''}
                        </span>
                      </div>

                      {/* Keywords */}
                      <div className="flex flex-wrap gap-1 mt-2">
                        {topic.keywords.slice(0, 4).map((keyword, i) => (
                          <span
                            key={i}
                            className="text-xs px-1.5 py-0.5 rounded"
                            style={{
                              backgroundColor: `${color}20`,
                              color: color,
                            }}
                          >
                            {keyword}
                          </span>
                        ))}
                        {topic.keywords.length > 4 && (
                          <span className="text-xs text-slate-500">
                            +{topic.keywords.length - 4}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </button>

                {/* Expand button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedTopic(isExpanded ? null : topic.id);
                  }}
                  className="w-full px-3 py-1.5 border-t border-slate-600/50 text-xs text-slate-400 hover:text-slate-300 hover:bg-slate-700/50 transition-colors flex items-center justify-center gap-1"
                >
                  {isExpanded ? 'Hide' : 'Show'} notes
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className={`h-3 w-3 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>

                {/* Expanded notes list */}
                {isExpanded && (
                  <div className="px-3 pb-3 border-t border-slate-600/50">
                    <div className="mt-2 space-y-1 max-h-40 overflow-y-auto">
                      {topic.note_titles.map((title, i) => {
                        const isCentral = topic.central_notes.includes(topic.note_ids[i]);
                        return (
                          <div
                            key={topic.note_ids[i]}
                            className="text-xs text-slate-300 py-1 px-2 rounded bg-slate-800/50 flex items-center gap-2"
                          >
                            {isCentral && (
                              <span title="Central note" className="text-amber-400">
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  className="h-3 w-3"
                                  fill="currentColor"
                                  viewBox="0 0 20 20"
                                >
                                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                </svg>
                              </span>
                            )}
                            <span className="truncate">{title}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Empty state */}
      {!loading && topics.length === 0 && (
        <div className="p-6 text-center">
          <div className="text-slate-500 mb-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-10 w-10 mx-auto"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
              />
            </svg>
          </div>
          <p className="text-slate-400 text-sm">No topics detected</p>
          <p className="text-slate-500 text-xs mt-1">
            Load a graph with connected notes to see topics
          </p>
        </div>
      )}

      {/* Orphan notes summary */}
      {!loading && orphanNotes.length > 0 && (
        <div className="p-3 border-t border-slate-700 shrink-0">
          <p className="text-xs text-slate-500">
            {orphanNotes.length} uncategorized note{orphanNotes.length !== 1 ? 's' : ''}
          </p>
        </div>
      )}
    </div>
  );
}

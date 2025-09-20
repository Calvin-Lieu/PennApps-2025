import React from 'react';
import TagSelector from './TagSelector';

interface AdvancedSettingsProps {
  selectedTags: { [key: string]: number };
  setSelectedTags: (tags: { [key: string]: number }) => void;
  tagMatchRule: 'any' | 'all';
  setTagMatchRule: (rule: 'any' | 'all') => void;
}

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({
  selectedTags,
  setSelectedTags,
  tagMatchRule,
  setTagMatchRule,
}) => {
  return (
    <div className="advanced-settings" style={{ borderTop: '1px solid #eee', paddingTop: '10px', marginTop: '10px' }}>
      <h4>Advanced Settings</h4>
      <TagSelector
        selectedTags={selectedTags}
        setSelectedTags={setSelectedTags}
      />
      <div style={{ marginTop: '10px' }}>
        <label>Tag Matching Rule: </label>
        <select
          value={tagMatchRule}
          onChange={(e) => setTagMatchRule(e.target.value as 'any' | 'all')}
          style={{ width: '100%', padding: '4px' }}
        >
          <option value="any">Any</option>
          <option value="all">All</option>
        </select>
      </div>
    </div>
  );
};

export default AdvancedSettings;

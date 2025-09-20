import React from 'react';

interface TagSelectorProps {
  selectedTags: { [key: string]: number };
  setSelectedTags: (tags: { [key: string]: number }) => void;
}

const availableTags = [
  'convenience',
  'supermarket',
  'mall',
  'cafe',
  'restaurant',
  'pub',
  'bar',
];

const TagSelector: React.FC<TagSelectorProps> = ({ selectedTags, setSelectedTags }) => {
  const handleTagCountChange = (tag: string, value: string) => {
    const num = parseInt(value, 10);
    const newSelectedTags = { ...selectedTags };

    if (isNaN(num) || num <= 0) {
      delete newSelectedTags[tag];
    } else {
      newSelectedTags[tag] = Math.min(num, 5); // Enforce max
    }
    setSelectedTags(newSelectedTags);
  };

  return (
    <div style={{ padding: '10px', border: '1px solid #ccc', borderRadius: '5px', marginTop: '10px' }}>
      <h4>Select Number of Waypoints for Each Type</h4>
      {availableTags.map(tag => (
        <div key={tag} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
          <label htmlFor={`count-${tag}`}>{tag}</label>
          <input
            type="number"
            id={`count-${tag}`}
            min={0}
            max={5}
            value={selectedTags[tag] || ''}
            onChange={(e) => handleTagCountChange(tag, e.target.value)}
            style={{ width: '60px', padding: '4px' }}
          />
        </div>
      ))}
    </div>
  );
};

export default TagSelector;

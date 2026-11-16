export const sectionIds = ['computation', 'learning', 'internet'] as const;
export const sections = {
  computation: { title: 'Computation', description: 'Small tools, code, and experiments that make an idea tangible.' },
  learning: { title: 'Learning', description: 'Reading notes, research habits, and questions worth keeping open.' },
  internet: { title: 'Internet', description: 'Personal websites, connected ideas, and life on the web.' },
};
export const tagUrl = (tag: string) => `/tags/${encodeURIComponent(tag)}/`;

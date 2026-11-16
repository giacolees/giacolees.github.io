import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';
import { sectionIds } from './topics';
const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({ title: z.string(), date: z.coerce.date(), updated: z.coerce.date().optional(), section: z.enum(sectionIds), description: z.string(), tags: z.array(z.string()).optional(), sample: z.boolean().default(false) }).refine(data => !data.updated || data.updated >= data.date, { message: "Updated date must be on or after publication", path: ["updated"] }),
});
export const collections = { posts };

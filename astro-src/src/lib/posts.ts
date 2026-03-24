export type Project = 'language' | 'stillness' | 'apokalyi';

export interface PostFrontmatter {
  title: string;
  date: string;
  excerpt?: string;
  project: Project;
  readingTime?: number;
}

export interface Post {
  slug: string;
  frontmatter: PostFrontmatter;
  url: string;
}

export const projectLabels: Record<Project, string> = {
  language: 'Language',
  stillness: 'Stillness',
  apokalyi: 'ApokalYi',
};

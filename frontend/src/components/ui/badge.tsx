import { cva, type VariantProps } from 'class-variance-authority';
import type { ComponentProps } from 'react';
import { cn } from '@/lib/utils';

const badgeVariants = cva('inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap', {
  variants: {
    variant: {
      default: 'border-transparent bg-primary/12 text-primary',
      secondary: 'border-transparent bg-muted text-muted-foreground',
      success: 'border-transparent bg-success/14 text-success',
      warning: 'border-transparent bg-warning/16 text-warning',
      destructive: 'border-transparent bg-destructive/12 text-destructive',
      info: 'border-transparent bg-info/14 text-info',
      outline: 'text-foreground',
    },
  },
  defaultVariants: { variant: 'default' },
});

export function Badge({ className, variant, ...props }: ComponentProps<'span'> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

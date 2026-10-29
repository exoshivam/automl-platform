import { Camera, Library, Upload } from 'lucide-react';
import { useProject } from '@/api/hooks';
import { Card } from '@/components/ui/card';
import { DemoBadge, Skeleton } from '@/components/ui/misc';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CameraCapture } from './camera-capture';
import { ImageUpload } from './image-upload';
import { StarterLibrary } from './starter-library';
import { TableUpload } from './table-upload';

/** PRD §5 step 3: the data screen adapts to the category; image tasks also get Capture. */
export function DataPage({ projectId }: { projectId: string }) {
  const { data: p } = useProject(projectId);
  if (!p) return <Skeleton className="h-80 rounded-xl" />;
  if (!p.dataset_id) return <Card className="p-6 text-sm text-destructive">This project has no dataset yet.</Card>;
  const image = p.task_type === 'image_classification' || p.task_type === 'object_detection';

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-lg font-semibold">Add data</h2>
        <DemoBadge group="data" />
        <p className="w-full text-sm text-muted-foreground">
          Files go straight to storage. New data collects in a draft; seal it on the Dataset tab to create a version you can train on.
        </p>
      </div>
      <Tabs defaultValue="upload">
        <TabsList>
          <TabsTrigger value="upload">
            <Upload /> Upload
          </TabsTrigger>
          {image && (
            <TabsTrigger value="camera">
              <Camera /> Camera
            </TabsTrigger>
          )}
          <TabsTrigger value="starters">
            <Library /> Starter datasets
          </TabsTrigger>
        </TabsList>
        <TabsContent value="upload">{image ? <ImageUpload project={p} /> : <TableUpload project={p} />}</TabsContent>
        {image && (
          <TabsContent value="camera">
            <CameraCapture project={p} />
          </TabsContent>
        )}
        <TabsContent value="starters">
          <StarterLibrary project={p} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

import { createRootRoute, createRoute, createRouter, Link, Outlet, redirect } from '@tanstack/react-router';
import type { ReactNode } from 'react';
import { useAuth } from '@/lib/auth';
import { AppShell } from '@/features/shell/app-shell';
import { LoginPage, RegisterPage, VerifyEmailPage } from '@/features/auth/auth-pages';
import { ProjectsPage } from '@/features/projects/projects-page';
import { NewProjectPage } from '@/features/projects/new-project-page';
import { ProjectPage } from '@/features/projects/project-page';
import { ProjectLayout } from '@/features/projects/project-layout';
import { DataPage } from '@/features/data/data-page';
import { DatasetPage, type DatasetSearch } from '@/features/data/dataset-page';
import { LabelPage } from '@/features/label/label-page';
import { TrainPage } from '@/features/ml/train-page';
import { EvaluatePage } from '@/features/ml/evaluate-page';
import { PlaygroundPage } from '@/features/ml/playground-page';
import { ExportPage } from '@/features/ml/export-page';
import { PeoplePage } from '@/features/face/people-page';
import { ExamplesPage } from '@/features/text/examples-page';
import { SpeakersPage } from '@/features/speaker/speakers-page';
import { DeployPage } from '@/features/deploy/deploy-page';
import { ApiKeysPage } from '@/features/deploy/api-keys-page';
import { SettingsPage } from '@/features/projects/settings-page';

const rootRoute = createRootRoute({
  component: Outlet,
  notFoundComponent: () => (
    <div className="grid min-h-dvh place-items-center p-6 text-center">
      <div>
        <p className="text-sm font-semibold text-primary">404</p>
        <h1 className="mt-1 text-2xl font-semibold">Page not found</h1>
        <Link to="/projects" className="mt-4 inline-block text-sm text-primary hover:underline">
          Back to projects
        </Link>
      </div>
    </div>
  ),
});

const guestOnly = () => {
  if (useAuth.getState().status === 'authenticated') throw redirect({ to: '/projects' });
};

const loginRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/login',
  validateSearch: (s: Record<string, unknown>): { redirect?: string } => (typeof s.redirect === 'string' ? { redirect: s.redirect } : {}),
  beforeLoad: guestOnly,
  component: LoginPage,
});
const registerRoute = createRoute({ getParentRoute: () => rootRoute, path: '/register', beforeLoad: guestOnly, component: RegisterPage });
const verifyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/verify-email',
  validateSearch: (s: Record<string, unknown>): { token?: string } => (typeof s.token === 'string' ? { token: s.token } : {}),
  component: VerifyEmailPage,
});

/** Everything under the shell needs a session; the access token is restored from the refresh cookie at boot. */
const appRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: 'app',
  beforeLoad: ({ location }) => {
    if (useAuth.getState().status !== 'authenticated') throw redirect({ to: '/login', search: { redirect: location.href } });
  },
  component: AppShell,
});

const indexRoute = createRoute({ getParentRoute: () => appRoute, path: '/', beforeLoad: () => { throw redirect({ to: '/projects' }); } });
const projectsRoute = createRoute({ getParentRoute: () => appRoute, path: '/projects', component: ProjectsPage });
const newProjectRoute = createRoute({ getParentRoute: () => appRoute, path: '/projects/new', component: NewProjectPage });
const apiKeysRoute = createRoute({ getParentRoute: () => appRoute, path: '/keys', component: ApiKeysPage });
const projectRoute = createRoute({
  getParentRoute: () => appRoute,
  path: '/projects/$projectId',
  component: function ProjectLayoutRoute() {
    const { projectId } = projectRoute.useParams();
    return <ProjectLayout projectId={projectId} />;
  },
});
const projectIndexRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/',
  component: function ProjectIndexRoute() {
    const { projectId } = projectRoute.useParams();
    return <ProjectPage projectId={projectId} />;
  },
});
const projectDataRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/data',
  component: function ProjectDataRoute() {
    const { projectId } = projectRoute.useParams();
    return <DataPage projectId={projectId} />;
  },
});
const str = (v: unknown) => (typeof v === 'string' && v ? v : undefined);
const projectDatasetRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/dataset',
  validateSearch: (s: Record<string, unknown>): DatasetSearch => ({
    v: str(s.v),
    tab: s.tab === 'validation' || s.tab === 'stats' || s.tab === 'versions' ? s.tab : undefined,
    label: str(s.label),
    split: s.split === 'train' || s.split === 'val' || s.split === 'test' ? s.split : undefined,
    flag: str(s.flag),
    unlabeled: s.unlabeled === true || s.unlabeled === 'true' ? true : undefined,
    select: s.select === true || s.select === 'true' ? true : undefined,
  }),
  component: function ProjectDatasetRoute() {
    const { projectId } = projectRoute.useParams();
    return <DatasetPage projectId={projectId} search={projectDatasetRoute.useSearch()} />;
  },
});
const projectLabelRoute = createRoute({
  getParentRoute: () => projectRoute,
  path: '/label',
  component: function ProjectLabelRoute() {
    const { projectId } = projectRoute.useParams();
    return <LabelPage projectId={projectId} />;
  },
});

const mlRoute = <P extends string>(path: P, Page: (p: { projectId: string }) => ReactNode) =>
  createRoute({
    getParentRoute: () => projectRoute,
    path,
    component: function ProjectMlRoute() {
      return <Page projectId={projectRoute.useParams().projectId} />;
    },
  });
const projectTrainRoute = mlRoute('/train', TrainPage);
const projectEvaluateRoute = mlRoute('/evaluate', EvaluatePage);
const projectPlaygroundRoute = mlRoute('/playground', PlaygroundPage);
const projectExportRoute = mlRoute('/export', ExportPage);
const projectPeopleRoute = mlRoute('/people', PeoplePage);
const projectExamplesRoute = mlRoute('/examples', ExamplesPage);
const projectSpeakersRoute = mlRoute('/speakers', SpeakersPage);
const projectDeployRoute = mlRoute('/deploy', DeployPage);
const projectFaceSettingsRoute = mlRoute('/settings', SettingsPage);

const routeTree = rootRoute.addChildren([loginRoute, registerRoute, verifyRoute, appRoute.addChildren([indexRoute, projectsRoute, newProjectRoute, apiKeysRoute, projectRoute.addChildren([projectIndexRoute, projectDataRoute, projectDatasetRoute, projectLabelRoute, projectTrainRoute, projectEvaluateRoute, projectPlaygroundRoute, projectExportRoute, projectPeopleRoute, projectFaceSettingsRoute, projectExamplesRoute, projectSpeakersRoute, projectDeployRoute])])]);

export const router = createRouter({ routeTree, defaultPreload: 'intent' });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

import { expect, test } from '@playwright/test';

// Against the MSW mocks (VITE_MOCKS=all). Demo account from src/mocks/db.ts.
test('log in, create an object detection project from a category', async ({ page }) => {
  await page.goto('/login');
  await page.getByLabel('Email').fill('demo@automl.dev');
  await page.getByLabel('Password').fill('demo1234');
  await page.getByRole('button', { name: 'Log in' }).click();
  await expect(page.getByRole('heading', { name: 'Projects' })).toBeVisible();

  await page.getByRole('link', { name: 'New project' }).first().click();
  await page.getByRole('button', { name: /Object detection/ }).click();
  await page.getByLabel('Project name').fill('E2E safety gear');
  await page.getByLabel('Class names').fill('helmet, vest');
  await page.getByLabel('Class names').press('Enter');
  await page.getByRole('button', { name: /Next: task spec/ }).click();
  await page.getByRole('tab', { name: /Raw JSON/ }).click();
  await expect(page.getByText('Valid task spec')).toBeVisible();
  await page.getByRole('button', { name: /Create project/ }).click();
  await expect(page.getByRole('heading', { name: 'E2E safety gear' })).toBeVisible();
});

test('describe path falls back to categories when the LLM is down', async ({ page }) => {
  await page.goto('/login');
  await page.getByLabel('Email').fill('demo@automl.dev');
  await page.getByLabel('Password').fill('demo1234');
  await page.getByRole('button', { name: 'Log in' }).click();
  await page.goto('/projects/new');
  await page.getByRole('button', { name: /Describe it/ }).click();
  await page.getByLabel('What should the model do?').fill('Sort my product photos into categories [llm-down]');
  await page.getByRole('button', { name: /Draft my project/ }).click();
  await expect(page.getByText(/AI helper isn't available/)).toBeVisible();
});

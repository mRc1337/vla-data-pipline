import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState, type ComponentProps, type ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CameraVideo,
  VideoPreviewErrorBoundary,
  type BrowserVideoMetadata,
} from "./videoPreview";

afterEach(cleanup);

function setMetadata(video: HTMLVideoElement, metadata: BrowserVideoMetadata): void {
  Object.defineProperty(video, "videoWidth", { configurable: true, value: metadata.width });
  Object.defineProperty(video, "videoHeight", { configurable: true, value: metadata.height });
  Object.defineProperty(video, "duration", { configurable: true, value: metadata.duration });
}

function cameraProps(camera: string, src: string, overrides: Partial<ComponentProps<typeof CameraVideo>> = {}) {
  return {
    camera,
    src,
    target: 0,
    retryToken: 0,
    onClick: vi.fn(),
    onLoadedMetadata: vi.fn(),
    onWaiting: vi.fn(),
    onStalled: vi.fn(),
    onCanPlay: vi.fn(),
    onError: vi.fn(),
    onRetry: vi.fn(),
    setRef: vi.fn(),
    ...overrides,
  };
}

describe("CameraVideo", () => {
  it("captures loadedmetadata values synchronously before the callback returns", () => {
    const onLoadedMetadata = vi.fn();
    const props = cameraProps("front", "/videos/front.mp4", { target: 2, onLoadedMetadata });
    render(<CameraVideo {...props} />);
    const video = document.querySelector("video") as HTMLVideoElement;
    setMetadata(video, { width: 1920, height: 1080, duration: 12.5 });

    fireEvent.loadedMetadata(video);

    expect(onLoadedMetadata).toHaveBeenCalledWith("front", video, {
      width: 1920, height: 1080, duration: 12.5,
    });
    expect(video.currentTime).toBe(2);
  });

  it("handles multiple cameras independently", () => {
    const metadata = vi.fn();
    render(<>
      <CameraVideo {...cameraProps("front", "/proxy/front.mp4", { onLoadedMetadata: metadata })} />
      <CameraVideo {...cameraProps("wrist", "/proxy/wrist.mp4", { onLoadedMetadata: metadata })} />
    </>);
    const videos = Array.from(document.querySelectorAll("video"));
    videos.forEach((video, index) => {
      setMetadata(video, { width: 640 + index, height: 480, duration: 4 });
      fireEvent.loadedMetadata(video);
    });

    expect(metadata).toHaveBeenCalledTimes(2);
    expect(metadata.mock.calls.map(([camera]) => camera)).toEqual(["front", "wrist"]);
  });

  it("replaces the element when an Episode or proxy source changes", () => {
    const props = cameraProps("front", "/episodes/0.mp4");
    const { rerender } = render(<CameraVideo key="episode-0" {...props} />);
    const firstVideo = document.querySelector("video");
    rerender(<CameraVideo key="episode-1" {...props} src="/episodes/1.mp4" />);
    const secondVideo = document.querySelector("video");
    expect(secondVideo).not.toBe(firstVideo);
    expect(secondVideo?.getAttribute("src")).toContain("/episodes/1.mp4");

    rerender(<CameraVideo key="proxy-episode-1" {...props} src="/proxy/episode-1-front.mp4" retryToken={1} />);
    expect(document.querySelector("video")?.getAttribute("src")).toContain("/proxy/");
  });

  it("shows a retryable camera-local error without removing other cameras", () => {
    const onError = vi.fn();
    const onRetry = vi.fn();
    const { rerender } = render(<>
      <CameraVideo {...cameraProps("front", "/front.mp4", { onError })} />
      <CameraVideo {...cameraProps("wrist", "/wrist.mp4")} />
    </>);
    const front = document.querySelector("video[data-camera='front']") as HTMLVideoElement;
    fireEvent(front, new Event("error", { bubbles: true }));
    expect(onError).toHaveBeenCalledWith("front", front, expect.stringContaining("front"));

    rerender(<>
      <CameraVideo {...cameraProps("front", "/front.mp4", { error: "源视频不可读", onError, onRetry })} />
      <CameraVideo {...cameraProps("wrist", "/wrist.mp4")} />
    </>);
    expect(screen.getAllByRole("alert")[0].textContent).toContain("源视频不可读");
    expect(document.querySelector("video[data-camera='wrist']")).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "重试该摄像头" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

function ThrowingPreview(): ReactElement {
  throw new Error("preview render failed");
}

function RecoverablePreview(): ReactElement {
  const [broken, setBroken] = useState(true);
  return <VideoPreviewErrorBoundary onRetry={() => setBroken(false)}>
    {broken ? <ThrowingPreview /> : <div>预览已恢复</div>}
  </VideoPreviewErrorBoundary>;
}

describe("VideoPreviewErrorBoundary", () => {
  it("keeps the surrounding page usable and reloads the preview after an error", () => {
    render(<div data-testid="page"><RecoverablePreview /></div>);
    expect(screen.getByTestId("page")).not.toBeNull();
    expect(screen.getByText("视频预览区域发生异常")).not.toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "重新加载预览" }));
    expect(screen.getByText("预览已恢复")).not.toBeNull();
    expect(screen.getByTestId("page")).not.toBeNull();
  });
});

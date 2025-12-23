package elementfinder.agent;

import elementfinder.core.ElementScanner;
import org.json.JSONObject;

import javax.imageio.ImageIO;
import java.awt.AWTException;
import java.awt.GraphicsConfiguration;
import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.instrument.Instrumentation;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Base64;

/**
 * Java agent that attaches to a JVM, discovers Swing/JavaFX components and writes snapshot files.
 */
public final class ElementFinderAgent {

    private static final DateTimeFormatter TIMESTAMP_FORMATTER = 
        DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS").withZone(ZoneId.systemDefault());

    private ElementFinderAgent() {
    }

    /**
     * Returns the default log directory path (jvm/logs relative to current working directory).
     */
    public static Path defaultLogDirectory() {
        return Paths.get(System.getProperty("user.dir"), "jvm", "logs");
    }

    public static void premain(String agentArgs, Instrumentation inst) {
        execute(agentArgs);
    }

    public static void agentmain(String agentArgs, Instrumentation inst) {
        execute(agentArgs);
    }

    public static void main(String[] args) {
        String targetPath = args != null && args.length > 0 ? args[0] : null;
        execute(targetPath);
    }

    public static void execute(String agentArgs) {
        Instant now = Instant.now();
        try {
            // Parse log directory from agent args
            Path logDir = (agentArgs != null && !agentArgs.isBlank()) 
                ? Paths.get(agentArgs.trim()) 
                : defaultLogDirectory();

            // Ensure log directory exists
            Files.createDirectories(logDir);

            ElementScanner scanner = new ElementScanner();
            JSONObject snapshot = scanner.captureSnapshot();
            snapshot.put("generatedAt", now.toString());

            // Capture screenshot
            byte[] screenshotBytes = captureScreenshot(scanner);
            if (screenshotBytes != null) {
                String base64 = Base64.getEncoder().encodeToString(screenshotBytes);
                snapshot.put("b64", base64);
            } else {
                snapshot.put("b64", "");
            }

            // Generate timestamp-based filename
            String timestamp = TIMESTAMP_FORMATTER.format(now);
            String baseName = "snapshot-" + timestamp;
            Path jsonPath = logDir.resolve(baseName + ".json");
            Path pngPath = logDir.resolve(baseName + ".png");

            // Write JSON file
            String jsonContent = snapshot.toString();
            Files.writeString(jsonPath, jsonContent, StandardCharsets.UTF_8);

            // Write PNG file if screenshot captured
            if (screenshotBytes != null && screenshotBytes.length > 0) {
                Files.write(pngPath, screenshotBytes);
            }

            // Also output to stdout for debugging
            System.out.println("[ElementFinderAgent] Snapshot written to: " + jsonPath);

        } catch (Throwable ex) {
            System.err.println("[ElementFinderAgent] Failed to capture components: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private static byte[] captureScreenshot(ElementScanner scanner) {
        if (GraphicsEnvironment.isHeadless()) {
            System.err.println("[ElementFinderAgent] Headless environment detected, skipping screenshot capture.");
            return null;
        }
        try {
            Robot robot = new Robot();
            Rectangle applicationBounds = getApplicationBounds(scanner);
            Rectangle screenBounds = getFullScreenBounds();
            Rectangle captureBounds = applicationBounds;
            if (captureBounds == null || captureBounds.isEmpty()) {
                captureBounds = screenBounds;
            } else if (screenBounds != null && !screenBounds.isEmpty()) {
                Rectangle intersection = captureBounds.intersection(screenBounds);
                if (intersection.isEmpty()) {
                    captureBounds = screenBounds;
                } else {
                    captureBounds = intersection;
                }
            }
            if (captureBounds == null || captureBounds.isEmpty()) {
                System.err.println("[ElementFinderAgent] No visible bounds found for application windows.");
                return null;
            }
            BufferedImage image = robot.createScreenCapture(captureBounds);
            try (ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
                ImageIO.write(image, "png", outputStream);
                return outputStream.toByteArray();
            }
        } catch (SecurityException | AWTException | IOException ex) {
            System.err.println("[ElementFinderAgent] Unable to capture screenshot: " + ex.getMessage());
            return null;
        }
    }

    private static Rectangle getApplicationBounds(ElementScanner scanner) {
        Rectangle combined = null;
        for (Object component : scanner.getComponentMap().values()) {
            Rectangle bounds = null;
            if (component instanceof Window window) {
                bounds = getWindowBounds(window);
            } else if (component instanceof javafx.stage.Window fxWindow) {
                bounds = getFxWindowBounds(fxWindow);
            }
            if (bounds != null) {
                combined = combined == null ? new Rectangle(bounds) : combined.union(bounds);
            }
        }
        if (combined == null) {
            for (Window window : Window.getWindows()) {
                Rectangle bounds = getWindowBounds(window);
                if (bounds != null) {
                    combined = combined == null ? new Rectangle(bounds) : combined.union(bounds);
                }
            }
        }
        if (combined == null) {
            for (javafx.stage.Window fxWindow : javafx.stage.Window.getWindows()) {
                Rectangle bounds = getFxWindowBounds(fxWindow);
                if (bounds != null) {
                    combined = combined == null ? new Rectangle(bounds) : combined.union(bounds);
                }
            }
        }
        return combined;
    }

    private static Rectangle getFullScreenBounds() {
        GraphicsEnvironment environment = GraphicsEnvironment.getLocalGraphicsEnvironment();
        GraphicsDevice[] devices = environment.getScreenDevices();
        Rectangle combined = null;
        for (GraphicsDevice device : devices) {
            GraphicsConfiguration configuration = device.getDefaultConfiguration();
            Rectangle bounds = configuration.getBounds();
            if (combined == null) {
                combined = new Rectangle(bounds);
            } else {
                combined = combined.union(bounds);
            }
        }
        if (combined == null) {
            java.awt.Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
            combined = new Rectangle(0, 0, screenSize.width, screenSize.height);
        }
        return combined;
    }

    private static Rectangle getWindowBounds(Window window) {
        if (window == null || !window.isShowing() || !window.isDisplayable()) {
            return null;
        }
        try {
            java.awt.Point location = window.getLocationOnScreen();
            java.awt.Dimension size = window.getSize();
            if (location == null || size == null || size.width <= 0 || size.height <= 0) {
                return null;
            }
            return new Rectangle(location.x, location.y, size.width, size.height);
        } catch (java.awt.IllegalComponentStateException ex) {
            return null;
        }
    }

    private static Rectangle getFxWindowBounds(javafx.stage.Window window) {
        if (window == null || !window.isShowing()) {
            return null;
        }
        double width = window.getWidth();
        double height = window.getHeight();
        if (width <= 0 || height <= 0) {
            return null;
        }
        return new Rectangle(
                (int) Math.round(window.getX()),
                (int) Math.round(window.getY()),
                (int) Math.round(width),
                (int) Math.round(height)
        );
    }
}


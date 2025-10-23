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
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Base64;
import java.util.Locale;
import java.nio.file.attribute.FileTime;

/**
 * Java agent that attaches to a JVM, reuses ajan-parent discovery logic and emits the component tree as JSON.
 */
public final class ElementFinderAgent {

    private static final String LOG_DIRECTORY = "logs";
    private static final DateTimeFormatter FILE_TS = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.ROOT)
            .withZone(ZoneId.systemDefault());

    private ElementFinderAgent() {
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
        OutputContext context = null;
        try {
            context = resolveOutput(agentArgs, now);
            prepareLogDirectories(context);
            logDebug(context, now, "starting capture");

            ElementScanner scanner = new ElementScanner();
            JSONObject snapshot = scanner.captureSnapshot();
            snapshot.put("generatedAt", now.toString());

            writeSnapshot(context, snapshot, scanner);
        } catch (Throwable ex) {
            System.err.println("[ElementFinderAgent] Failed to capture components: " + ex.getMessage());
            ex.printStackTrace();
            if (context != null) {
                logDebug(context, now, "failure: " + summarize(ex));
                writeErrorDump(context.root(), now, ex);
            } else {
                Path fallback = defaultLogDirectory();
                try {
                    Files.createDirectories(fallback);
                    Path debugFile = fallback.resolve("agent-debug.log");
                    String message = now + " - failure before context: " + summarize(ex) + System.lineSeparator();
                    Files.writeString(debugFile, message, StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                    writeErrorDump(fallback, now, ex);
                } catch (IOException ignored) {
                    // swallow
                }
            }
        }
    }

    private static OutputContext resolveOutput(String agentArgs, Instant timestamp) {
        String baseName = "snapshot-" + FILE_TS.format(timestamp);
        Path defaultDirectory = null;
        if (agentArgs != null && !agentArgs.isBlank()) {
            try {
                Path candidate = Paths.get(agentArgs.trim());
                if (!candidate.isAbsolute()) {
                    candidate = Paths.get(System.getProperty("user.dir")).resolve(candidate);
                }
                if (Files.isDirectory(candidate)) {
                    Path root = candidate.toAbsolutePath();
                    return buildDirectoryContext(root, baseName);
                }
                Path jsonFile = candidate.toAbsolutePath();
                return buildFileContext(jsonFile, defaultLogDirectory(), baseName);
            } catch (InvalidPathException ex) {
                System.err.println("[ElementFinderAgent] Invalid path provided, falling back to default log directory: "
                        + ex.getMessage());
            }
        }
        if (defaultDirectory == null) {
            defaultDirectory = defaultLogDirectory();
        }
        Path root = defaultDirectory.toAbsolutePath();
        return buildDirectoryContext(root, baseName);
    }

    private static OutputContext buildDirectoryContext(Path root, String baseName) {
        Path jsonDir = root;
        Path screenshotDir = root;
        Path jsonFile = jsonDir.resolve(baseName + ".json");
        Path screenshotFile = screenshotDir.resolve(baseName + ".png");
        return new OutputContext(root, jsonDir, screenshotDir, jsonFile, screenshotFile);
    }

    private static OutputContext buildFileContext(Path jsonFile, Path defaultDirectory, String fallbackBaseName) {
        Path root = jsonFile.getParent();
        if (root == null) {
            root = defaultDirectory;
        }
        Path jsonDir = jsonFile.getParent();
        if (jsonDir == null) {
            jsonDir = root;
        }
        String baseName = stripExtension(jsonFile.getFileName().toString());
        if (baseName.isBlank()) {
            baseName = fallbackBaseName;
        }
        Path screenshotFile = root.resolve(baseName + ".png");
        Path screenshotDir = screenshotFile.getParent();
        if (screenshotDir == null) {
            screenshotDir = root;
        }
        return new OutputContext(root, jsonDir, screenshotDir, jsonFile, screenshotFile);
    }

    private static void writeSnapshot(OutputContext context, JSONObject snapshot, ElementScanner scanner) throws IOException {
        logDebug(context, Instant.now(), "capturing snapshot to " + context.jsonPath());

        Path jsonOutput = context.jsonPath();
        Path screenshotOutput = context.screenshotPath();
        byte[] screenshotBytes = captureScreenshot(scanner);
        if (screenshotBytes != null) {
            String base64 = Base64.getEncoder().encodeToString(screenshotBytes);
            snapshot.put("b64", base64);
            Files.write(screenshotOutput, screenshotBytes);
            System.out.println("[ElementFinderAgent] Screenshot written to: " + screenshotOutput);
        } else {
            snapshot.put("b64", "");
            System.err.println("[ElementFinderAgent] Screenshot capture returned no data.");
        }

        Files.writeString(jsonOutput, snapshot.toString(2), StandardCharsets.UTF_8);
        System.out.println("[ElementFinderAgent] Component snapshot written to: " + jsonOutput);
    }

    private static void prepareLogDirectories(OutputContext context) throws IOException {
        Files.createDirectories(context.jsonDirectory());
        Files.createDirectories(context.screenshotDirectory());
    }

    private static void logDebug(OutputContext context, Instant timestamp, String message) {
        try {
            Path debugFile = context.root().resolve("agent-debug.log");
            String debugLine = timestamp + " - " + message + System.lineSeparator();
            Files.writeString(debugFile, debugLine, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException debugEx) {
            System.err.println("[ElementFinderAgent] Failed to update debug log: " + debugEx.getMessage());
        }
    }

    private static void writeErrorDump(Path root, Instant timestamp, Throwable ex) {
        try {
            FileTime fileTime = FileTime.from(timestamp);
            String baseName = "agent-error-" + FILE_TS.format(timestamp) + ".log";
            Path errorFile = root.resolve(baseName);
            Files.createDirectories(root);
            try (StringWriter sw = new StringWriter(); PrintWriter pw = new PrintWriter(sw)) {
                ex.printStackTrace(pw);
                pw.flush();
                Files.writeString(errorFile, sw.toString(), StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
                Files.setLastModifiedTime(errorFile, fileTime);
            }
        } catch (IOException ignored) {
            // skip
        }
    }

    private static String summarize(Throwable ex) {
        return ex.getClass().getSimpleName() + ": " + ex.getMessage();
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

    public static Path defaultLogDirectory() {
        try {
            var protectionDomain = ElementFinderAgent.class.getProtectionDomain();
            if (protectionDomain != null) {
                var codeSource = protectionDomain.getCodeSource();
                if (codeSource != null && codeSource.getLocation() != null) {
                    Path location = Paths.get(codeSource.getLocation().toURI());
                    Path base = Files.isDirectory(location) ? location : location.getParent();
                    Path projectRoot = base != null ? base.getParent() : null;
                    if (projectRoot != null && "target".equalsIgnoreCase(projectRoot.getFileName().toString())) {
                        projectRoot = projectRoot.getParent();
                    }
                    if (projectRoot != null) {
                        return projectRoot.resolve(LOG_DIRECTORY);
                    } else if (base != null) {
                        return base.resolve(LOG_DIRECTORY);
                    }
                }
            }
        } catch (URISyntaxException | SecurityException ignored) {
        }
        Path workingDir = Paths.get(System.getProperty("user.dir"));
        if (Files.isDirectory(workingDir)) {
            return workingDir.resolve(LOG_DIRECTORY);
        }
        return Paths.get(System.getProperty("user.home")).resolve(LOG_DIRECTORY);
    }

    private static String stripExtension(String fileName) {
        int lastDot = fileName.lastIndexOf('.');
        if (lastDot > 0) {
            return fileName.substring(0, lastDot);
        }
        return fileName;
    }

    private record OutputContext(
            Path root,
            Path jsonDirectory,
            Path screenshotDirectory,
            Path jsonPath,
            Path screenshotPath) {
    }
}

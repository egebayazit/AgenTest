package elementfinder.cli;

import com.sun.tools.attach.VirtualMachine;
import com.sun.tools.attach.VirtualMachineDescriptor;
import elementfinder.agent.ElementFinderAgent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;

/**
 * Lightweight console utility that mimics ajan-attacher to load the element finder agent into a target JVM.
 */
public final class AttachMain {

    private static final String DEFAULT_JAR = "jvm-element-finder-1.0-SNAPSHOT-jar-with-dependencies.jar";

    private AttachMain() {
    }

    public static void main(String[] args) {
        System.out.println("=== JVM Element Finder ===\n");

        List<VirtualMachineDescriptor> descriptors = VirtualMachine.list();
        if (descriptors.isEmpty()) {
            System.err.println("No attachable JVM processes detected.");
            return;
        }

        final String requestedPid = (args.length > 0 && !args[0].isBlank())
                ? args[0].trim()
                : null;

        // Second argument is the log directory (passed from C++ code)
        final String logDirArg = (args.length > 1 && !args[1].isBlank())
                ? args[1].trim()
                : null;

        VirtualMachineDescriptor descriptor = null;
        if (requestedPid != null) {
            Optional<VirtualMachineDescriptor> explicitTarget = descriptors.stream()
                    .filter(desc -> desc.id().equals(requestedPid))
                    .findFirst();
            if (explicitTarget.isPresent()) {
                descriptor = explicitTarget.get();
                System.out.printf("Attaching to PID %s (%s)%n", descriptor.id(), descriptor.displayName());
            } else {
                System.err.printf("PID %s is not an attachable JVM. Falling back to active window detection.%n", requestedPid);
            }
        }
        if (descriptor == null) {
            Optional<VirtualMachineDescriptor> target = ActiveJvmFinder.detect(descriptors);
            if (target.isEmpty()) {
                System.err.println("Unable to detect an active Java window. Focus the target application and retry.");
                System.err.println("Attachable JVMs:");
                descriptors.forEach(desc -> System.err.printf(" PID: %s\tDisplay: %s%n", desc.id(), desc.displayName()));
                System.exit(1);
                return;
            }
            descriptor = target.get();
            System.out.printf("Attaching to PID %s (%s)%n", descriptor.id(), descriptor.displayName());
        }

        Path agentJar = resolveAgentJar();
        if (!Files.exists(agentJar)) {
            System.err.println("Agent jar not found at " + agentJar);
            System.err.println("Run `mvn package` in this directory first.");
            return;
        }

        // Use log directory from argument if provided, otherwise use default
        Path logDirectory = (logDirArg != null)
                ? Paths.get(logDirArg).toAbsolutePath()
                : ElementFinderAgent.defaultLogDirectory().toAbsolutePath();
        try {
            Files.createDirectories(logDirectory);
        } catch (IOException e) {
            System.err.println("Warning: unable to prepare log directory " + logDirectory + ": " + e.getMessage());
        }
        try {
            VirtualMachine vm = VirtualMachine.attach(descriptor);
            try {
                vm.loadAgent(agentJar.toString(), logDirectory.toString());
                System.out.println("Agent successfully loaded.");
                printLogLocation(logDirectory);
            } finally {
                vm.detach();
            }
        } catch (Exception e) {
            System.err.println("Failed to attach agent: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static Path resolveAgentJar() {
        Path targetDir = Paths.get(System.getProperty("user.dir"), "target");
        Path jarPath = targetDir.resolve(DEFAULT_JAR);
        if (Files.exists(jarPath)) {
            return jarPath;
        }

        Optional<Path> candidate = findJar(targetDir);
        return candidate.orElse(jarPath);
    }

    private static Optional<Path> findJar(Path targetDir) {
        if (!Files.isDirectory(targetDir)) {
            return Optional.empty();
        }
        try {
            return Files.list(targetDir)
                    .filter(p -> p.getFileName().toString().endsWith("jar-with-dependencies.jar"))
                    .findFirst();
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    private static void printLogLocation(Path logs) {
        System.out.println("Logs (JSON/PNG) will appear under: " + logs);
        if (!Files.isDirectory(logs)) {
            System.out.println("Directory will be created on first snapshot.");
        }
    }
}

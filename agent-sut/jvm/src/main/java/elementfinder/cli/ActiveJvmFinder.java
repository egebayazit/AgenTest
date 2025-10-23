package elementfinder.cli;

import com.sun.jna.Pointer;
import com.sun.jna.platform.win32.User32;
import com.sun.jna.platform.win32.WinDef;
import com.sun.jna.ptr.IntByReference;
import com.sun.tools.attach.VirtualMachineDescriptor;

import java.util.List;
import java.util.Optional;

final class ActiveJvmFinder {

    private ActiveJvmFinder() {
    }

    static Optional<VirtualMachineDescriptor> detect(List<VirtualMachineDescriptor> descriptors) {
        WinDef.HWND foreground = User32.INSTANCE.GetForegroundWindow();
        if (foreground == null || Pointer.nativeValue(foreground.getPointer()) == 0) {
            return Optional.empty();
        }

        WinDef.HWND root = User32.INSTANCE.GetAncestor(foreground, User32.GA_ROOTOWNER);
        WinDef.HWND window = (root != null && Pointer.nativeValue(root.getPointer()) != 0) ? root : foreground;

        IntByReference pidRef = new IntByReference();
        User32.INSTANCE.GetWindowThreadProcessId(window, pidRef);
        int pid = pidRef.getValue();
        if (pid <= 0) {
            return Optional.empty();
        }
        String pidString = Integer.toString(pid);

        return descriptors.stream()
                .filter(vm -> vm.id().equals(pidString))
                .findFirst();
    }
}

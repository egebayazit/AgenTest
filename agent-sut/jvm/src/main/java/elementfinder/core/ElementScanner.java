package elementfinder.core;

import javafx.application.Platform;
import javafx.collections.ObservableList;
import javafx.embed.swing.JFXPanel;
import javafx.embed.swing.SwingNode;
import javafx.geometry.Bounds;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.Labeled;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import org.json.JSONArray;
import org.json.JSONObject;

import javax.swing.AbstractButton;
import javax.swing.ComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLayeredPane;
import javax.swing.JLabel;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.text.JTextComponent;
import javax.swing.JToolBar;
import javax.swing.JTree;
import javax.swing.JViewport;
import javax.swing.MenuElement;
import javax.swing.RootPaneContainer;
import javax.swing.SwingUtilities;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.IllegalComponentStateException;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Window;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Reuses the component discovery flow from ajan-parent to snapshot Swing and JavaFX hierarchies.
 */
public class ElementScanner {

    private final Map<String, Object> componentMap = new LinkedHashMap<>();

    public JSONObject captureSnapshot() {
        componentMap.clear();

        ComponentNode root = new ComponentNode("root", "root", "root");

        int windowIndex = 0;
        Window[] windows = Window.getWindows();
        for (Window window : windows) {
            if (shouldSkipAwtWindow(window)) {
                windowIndex++;
                continue;
            }
            String id = "W" + windowIndex;
            ComponentNode windowNode = registerComponent(root, id, window);
            if (window instanceof Container container) {
                scanSwingContainer(container, id, windowNode);
            }
            windowIndex++;
        }

        List<javafx.stage.Window> fxWindows = javafx.stage.Window.getWindows();
        for (javafx.stage.Window fxWindow : fxWindows) {
            if (shouldSkipFxWindow(fxWindow)) {
                windowIndex++;
                continue;
            }
            String id = "W" + windowIndex;
            ComponentNode fxNode = registerComponent(root, id, fxWindow);
            if (fxWindow.getScene() != null) {
                scanFxObject(fxNode, id, fxWindow.getScene());
            }
            windowIndex++;
        }

        JSONArray elements = new JSONArray();
        for (ComponentNode child : root.getChildren()) {
            elements.put(child.toJson());
        }

        JSONObject result = new JSONObject();
        result.put("componentCount", componentMap.size());
        result.put("elements", elements);
        return result;
    }

    private void scanSwingContainer(Container container, String currentId, ComponentNode parentNode) {
        Component[] components = container.getComponents();
        for (int j = 0; j < components.length; j++) {
            Component child = components[j];
            int index = j;

            if (container instanceof JLayeredPane) {
                Object owningWindow = SwingUtilities.windowForComponent(child);
                if (owningWindow instanceof RootPaneContainer rootPaneContainer) {
                    if (child == rootPaneContainer.getContentPane()) {
                        index = 100;
                    } else if (child instanceof JMenuBar) {
                        index = 101;
                    }
                }
            }

            String newId = currentId + "." + index;
            ComponentNode childNode = registerComponent(parentNode, newId, child);

            if (child instanceof Container childContainer) {
                if (child instanceof JViewport viewport) {
                    handleViewport(viewport, newId, childNode);
                } else {
                    scanSwingContainer(childContainer, newId, childNode);
                }
            }

            if (child instanceof JFXPanel jfxPanel) {
                Scene scene = jfxPanel.getScene();
                if (scene != null) {
                    String sceneId = newId + ".0";
                    ComponentNode sceneNode = registerComponent(childNode, sceneId, scene);
                    scanFxObject(sceneNode, sceneId, scene);
                }
            }
        }
    }

    private void handleViewport(JViewport viewport, String currentId, ComponentNode parentNode) {
        Component view = viewport.getView();
        if (view == null) {
            return;
        }
        String viewId = currentId + ".0";
        ComponentNode viewNode = registerComponent(parentNode, viewId, view);
        if (view instanceof Container container) {
            scanSwingContainer(container, viewId, viewNode);
        }
    }

    private void scanFxObject(ComponentNode parentNode, String currentId, Object object) {
        if (object instanceof Stage stage) {
            Scene scene = stage.getScene();
            if (scene != null) {
                String sceneId = currentId + ".0";
                ComponentNode sceneNode = registerComponent(parentNode, sceneId, scene);
                scanFxObject(sceneNode, sceneId, scene);
            }
            return;
        }

        if (object instanceof Scene scene) {
            Node root = scene.getRoot();
            if (root != null) {
                String rootId = currentId + ".0";
                ComponentNode rootNode = registerComponent(parentNode, rootId, root);
                scanFxNode(rootNode, rootId, root);
            }
            return;
        }

        if (object instanceof Tab tab) {
            Node content = tab.getContent();
            if (content != null) {
                String contentId = currentId + ".0";
                ComponentNode contentNode = registerComponent(parentNode, contentId, content);
                scanFxNode(contentNode, contentId, content);
            }
            return;
        }

        if (object instanceof SwingNode swingNode) {
            JComponent content = swingNode.getContent();
            if (content != null) {
                String contentId = currentId + ".0";
                ComponentNode swingContentNode = registerComponent(parentNode, contentId, content);
                scanSwingContainer(content, contentId, swingContentNode);
            }
            return;
        }

        if (object instanceof Node node) {
            scanFxNode(parentNode, currentId, node);
        }
    }

    private void scanFxNode(ComponentNode parentNode, String currentId, Node node) {
        if (node instanceof Parent parent) {
            AtomicInteger counter = new AtomicInteger();

            if (parent instanceof TabPane tabPane) {
                for (Tab tab : tabPane.getTabs()) {
                    String tabId = currentId + "." + counter.getAndIncrement();
                    ComponentNode tabNode = registerComponent(parentNode, tabId, tab);
                    scanFxObject(tabNode, tabId, tab);
                }
            }

            ObservableList<Node> children = parent.getChildrenUnmodifiable();
            for (Node child : children) {
                String childId = currentId + "." + counter.getAndIncrement();
                ComponentNode childNode = registerComponent(parentNode, childId, child);
                scanFxNode(childNode, childId, child);
            }
        }
    }

    private ComponentNode registerComponent(ComponentNode parent, String id, Object component) {
        ComponentNode node = new ComponentNode(id, component.getClass().getSimpleName(), getComponentText(component));
        parent.addChild(node);
        componentMap.put(id, component);
        applyGeometry(node, component);
        return node;
    }

    private boolean shouldSkipAwtWindow(Window window) {
        if (!window.isDisplayable()) {
            return true;
        }
        if (window instanceof JDialog dialog && "AJAN TREE".equalsIgnoreCase(dialog.getTitle())) {
            return true;
        }
        String simpleName = window.getClass().getSimpleName();
        return simpleName.equalsIgnoreCase("SHAREDOWNERFRAME")
                || simpleName.equalsIgnoreCase("HEAVYWEIGHTWINDOW")
                || simpleName.equalsIgnoreCase("JLightweightFrame");
    }

    private boolean shouldSkipFxWindow(javafx.stage.Window window) {
        return window.getClass().getSimpleName().equalsIgnoreCase("EmbeddedWindow");
    }

    private String getComponentText(Object component) {
        if (component instanceof Component awt) {
            if (awt instanceof AbstractButton button) {
                return safe(button.getText());
            } else if (awt instanceof JLabel label) {
                return safe(label.getText());
            } else if (awt instanceof JTextComponent textComponent) {
                return safe(textComponent.getText());
            } else if (awt instanceof JPopupMenu popupMenu) {
                MenuElement[] elements = popupMenu.getSubElements();
                if (elements != null && elements.length > 0 && elements[0] instanceof JMenuItem menuItem) {
                    return safe(menuItem.getText());
                }
            } else if (awt instanceof JTree tree) {
                Object root = tree.getModel().getRoot();
                return root == null ? "" : safe(root.toString());
            } else if (awt instanceof JComboBox<?> comboBox) {
                ComboBoxModel<?> model = comboBox.getModel();
                if (model != null && model.getSize() > 0) {
                    Object value = comboBox.getSelectedItem();
                    if (value == null) {
                        value = model.getElementAt(0);
                    }
                    return value == null ? "" : safe(value.toString());
                }
            } else if (awt instanceof JFrame frame) {
                return safe(frame.getTitle());
            } else if (awt instanceof JDialog dialog) {
                return safe(dialog.getTitle());
            } else if (awt instanceof JTable table) {
                Dimension dimension = table.getPreferredSize();
                return "rows=" + table.getRowCount() + ", cols=" + table.getColumnCount() + ", size=" + dimension.width + "x" + dimension.height;
            }
        } else if (component instanceof Node node) {
            if (node instanceof Labeled labeled) {
                return safe(labeled.getText());
            } else if (node instanceof Text text) {
                return safe(text.getText());
            } else if (node instanceof TextField field) {
                return safe(field.getText());
            }
        } else if (component instanceof Stage stage) {
            String title = getNarxTitle(stage);
            if (title == null) {
                title = stage.getTitle();
            }
            return safe(title);
        } else if (component instanceof Tab tab) {
            return safe(tab.getText());
        } else if (component instanceof Scene scene) {
            return scene.getClass().getSimpleName();
        }
        return "";
    }

    private String safe(String value) {
        return value == null ? "" : value;
    }

    private String getNarxTitle(javafx.stage.Window window) {
        String simpleName = window.getClass().getSimpleName();
        if ("NarxStage".equals(simpleName) || "PEStage".equals(simpleName)) {
            Node root = window.getScene().getRoot();
            if (root instanceof Parent parent) {
                for (Node node : parent.getChildrenUnmodifiable()) {
                    if (node.getClass().getSimpleName().contains("WindowBar") && node instanceof Parent windowBar) {
                        for (Node child : windowBar.getChildrenUnmodifiable()) {
                            if ("XLabel".equals(child.getClass().getSimpleName()) && child instanceof Label label) {
                                return label.getText();
                            }
                        }
                    }
                }
            }
        }
        return null;
    }

    public Map<String, Object> getComponentMap() {
        return componentMap;
    }

    private void applyGeometry(ComponentNode node, Object component) {
        if (component instanceof Component awtComponent) {
            Rectangle bounds = getSwingScreenBounds(awtComponent);
            if (bounds != null) {
                node.setGeometry(
                        (double) bounds.getX(),
                        (double) bounds.getY(),
                        (double) bounds.getWidth(),
                        (double) bounds.getHeight()
                );
            }
        } else if (component instanceof javafx.stage.Window fxWindow) {
            if (fxWindow.isShowing()) {
                node.setGeometry(fxWindow.getX(), fxWindow.getY(), fxWindow.getWidth(), fxWindow.getHeight());
            }
        } else if (component instanceof Node fxNode) {
            Bounds bounds = getFxScreenBounds(fxNode);
            if (bounds != null) {
                node.setGeometry(bounds.getMinX(), bounds.getMinY(), bounds.getWidth(), bounds.getHeight());
            }
        }
    }

    private Rectangle getSwingScreenBounds(Component component) {
        if (!component.isShowing() && !(component instanceof Window)) {
            return null;
        }
        try {
            Point location = component.getLocationOnScreen();
            Dimension size = component.getSize();
            if (size == null) {
                return null;
            }
            return new Rectangle(location.x, location.y, size.width, size.height);
        } catch (IllegalComponentStateException ex) {
            return null;
        }
    }

    private Bounds getFxScreenBounds(Node node) {
        if (node.getScene() == null) {
            return null;
        }
        return callOnFxThread(() -> {
            Bounds bounds = node.localToScreen(node.getBoundsInLocal());
            if (bounds == null || bounds.getWidth() == 0 || bounds.getHeight() == 0) {
                return null;
            }
            return bounds;
        });
    }

    private <T> T callOnFxThread(Callable<T> callable) {
        if (Platform.isFxApplicationThread()) {
            try {
                return callable.call();
            } catch (Exception e) {
                return null;
            }
        }
        FutureTask<T> task = new FutureTask<>(callable);
        Platform.runLater(task);
        try {
            return task.get();
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

}


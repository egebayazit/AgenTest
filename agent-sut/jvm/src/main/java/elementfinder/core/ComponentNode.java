package elementfinder.core;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

class ComponentNode {

    private final String id;
    private final String type;
    private final String text;
    private final List<ComponentNode> children = new ArrayList<>();
    private Double x;
    private Double y;
    private Double width;
    private Double height;

    ComponentNode(String id, String type, String text) {
        this.id = Objects.requireNonNull(id, "id");
        this.type = type == null ? "" : type;
        this.text = text == null ? "" : text;
    }

    void addChild(ComponentNode child) {
        children.add(child);
    }

    void setGeometry(Double x, Double y, Double width, Double height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    List<ComponentNode> getChildren() {
        return children;
    }

    JSONObject toJson() {
        JSONObject json = new JSONObject();
        json.put("id", id);
        json.put("class", type);
        json.put("text", text);
        if (x != null) {
            json.put("x", x);
        }
        if (y != null) {
            json.put("y", y);
        }
        if (width != null) {
            json.put("width", width);
        }
        if (height != null) {
            json.put("height", height);
        }
        if (!children.isEmpty()) {
            JSONArray array = new JSONArray();
            for (ComponentNode child : children) {
                array.put(child.toJson());
            }
            json.put("children", array);
        }
        return json;
    }
}

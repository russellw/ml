package prover;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public final class Type {
    public static final Type BOOLEAN = new Type("boolean");
    public static final Type INDIVIDUAL = new Type("individual");
    public static final Type NUMBER = new Type("number");
    private final String name;
    public final Category[] categories;
    private final Map<String, Category> map;

    public Type(String name, Collection<String> categorical) {
        this.name = name;
        categories = new Category[categorical.size()];
        map = new HashMap<>();
        var i = 0;
        for (var s : categorical) {
            var category = new Category(this, s, i);
            categories[i] = category;
            map.put(s, category);
            i++;
        }
    }

    private Type(String name) {
        this.name = name;
        categories = null;
        map = null;
    }

    public Category category(String s) {
        return map.get(s);
    }

    @Override
    public String toString() {
        return name;
    }
}

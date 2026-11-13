package org.cvision.assist;

import java.net.URI;
import java.net.URISyntaxException;

/** URL policy for the bundled WebView, independent of Android for JVM tests. */
public final class BundlePolicy {
    public static final String LOCAL_HOST = "appassets.androidplatform.net";
    public static final String LOCAL_ORIGIN = "https://" + LOCAL_HOST;
    private static final String PRODUCT_HOST = "world.openfoodfacts.org";

    private BundlePolicy() {}

    /** Tests the origin only. Use assetPath before opening any bundled file. */
    public static boolean isLocal(String url) {
        return hasOrigin(parse(url), LOCAL_HOST);
    }

    /** Returns an APK asset path, or null for unknown or non-canonical paths. */
    public static String assetPath(String url) {
        URI uri = parse(url);
        if (!hasOrigin(uri, LOCAL_HOST)) return null;
        String path = uri.getRawPath();
        if (path == null || path.isEmpty()) path = "/";
        // Bundle names need no URL escapes. Reject all escapes before decoding,
        // including double-encoded dots, separators and NULs. Never normalize
        // traversal into a valid path or pass it to AssetManager.
        if (!canonicalPath(path)) return null;
        if (path.equals("/") || path.equals("/assist") || path.equals("/assist.html")) {
            return "web/assist.html";
        }
        if (path.startsWith("/static/") || path.startsWith("/licenses/")
                || path.equals("/THIRD_PARTY_NOTICES.txt")) {
            return "web" + path;
        }
        return null;
    }

    /** The sole optional network operation: a single EAN-13 product API read. */
    public static boolean isAllowedNetwork(String url) {
        URI uri = parse(url);
        return hasOrigin(uri, PRODUCT_HOST)
                && uri.getRawPath().matches("/api/v2/product/[0-9]{13}\\.json");
    }

    private static URI parse(String url) {
        if (url == null || url.length() > 8192) return null;
        try {
            return new URI(url);
        } catch (URISyntaxException ex) {
            return null;
        }
    }

    private static boolean hasOrigin(URI uri, String host) {
        if (uri == null || uri.isOpaque() || !"https".equalsIgnoreCase(uri.getScheme())
                || !host.equalsIgnoreCase(uri.getHost()) || uri.getRawUserInfo() != null) return false;
        // Checking the raw authority as well avoids accepting alternative port
        // spellings, percent-encoded hostnames, userinfo or trailing separators.
        String authority = uri.getRawAuthority();
        return host.equalsIgnoreCase(authority) || (host + ":443").equalsIgnoreCase(authority);
    }

    private static boolean canonicalPath(String path) {
        if (path.equals("/")) return true;
        if (!path.startsWith("/") || path.endsWith("/")) return false;
        for (String segment : path.substring(1).split("/", -1)) {
            if (segment.isEmpty() || segment.equals(".") || segment.equals("..")
                    || !segment.matches("[A-Za-z0-9_.-]+")) return false;
        }
        return true;
    }
}

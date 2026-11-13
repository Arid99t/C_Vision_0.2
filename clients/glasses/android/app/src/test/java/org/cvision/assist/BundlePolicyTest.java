package org.cvision.assist;

import org.junit.Test;

import static org.junit.Assert.*;

public class BundlePolicyTest {
    private static final String LOCAL = "https://appassets.androidplatform.net";
    private static final String PRODUCT = "https://world.openfoodfacts.org/api/v2/product/1234567890123.json";

    @Test public void mapsBundledEntryPointsAndKeepsQueryOutOfAssetNames() {
        for (String path : new String[] {"", "/", "/assist", "/assist.html?threads=1#start"}) {
            assertEquals(path, "web/assist.html", BundlePolicy.assetPath(LOCAL + path));
        }
        assertEquals("web/static/js/assist/app.js", BundlePolicy.assetPath(LOCAL + "/static/js/assist/app.js?v=1"));
        assertEquals("web/static/vendor/ort/ort-wasm-simd-threaded.wasm",
                BundlePolicy.assetPath(LOCAL + "/static/vendor/ort/ort-wasm-simd-threaded.wasm"));
        assertEquals("web/licenses/PaddleOCR.LICENSE.txt", BundlePolicy.assetPath(LOCAL + "/licenses/PaddleOCR.LICENSE.txt"));
        assertEquals("web/THIRD_PARTY_NOTICES.txt", BundlePolicy.assetPath(LOCAL + "/THIRD_PARTY_NOTICES.txt"));
        assertNull(BundlePolicy.assetPath(LOCAL + "/health"));
        assertNull(BundlePolicy.assetPath(LOCAL + "/recognize"));
        assertNull(BundlePolicy.assetPath(LOCAL + "/data/private.txt"));
    }

    @Test public void rejectsTraversalBeforeAnyFileIsOpened() {
        for (String path : new String[] {
                "/static/../private.txt", "/static/./js/app.js", "/static/js/../../private.txt",
                "/static/%2e%2e/private.txt", "/static/%2E%2E/private.txt", "/static/.%2e/private.txt",
                "/static/%252e%252e/private.txt", "/static/%2fetc/passwd", "/static/%5cprivate.txt",
                "/static/a%2Fb.js", "/static/a%255cb.js", "/static/..\\private.txt",
                "/static/abc%00.js", "/static/abc\u0000.js", "/static//js/app.js", "//static/js/app.js",
                "/static/", "/static", "/licenses/../assist.html", "/licenses/%2e%2e/assist.html",
                "/static/secret;file.js", "/static/colon:file.js", "/static/a b.js"}) {
            assertNull(path, BundlePolicy.assetPath(LOCAL + path));
        }
    }

    @Test public void trustsOnlyTheExactHttpsAppOrigin() {
        assertTrue(BundlePolicy.isLocal(LOCAL));
        assertTrue(BundlePolicy.isLocal(LOCAL + ":443/assist"));
        assertTrue(BundlePolicy.isLocal("HTTPS://APPASSETS.ANDROIDPLATFORM.NET/assist"));
        for (String url : new String[] {
                null, "", "/assist", "//appassets.androidplatform.net/assist",
                "http://appassets.androidplatform.net/assist", "file:///android_asset/web/assist.html",
                "content://appassets.androidplatform.net/assist", "javascript:alert(1)",
                "https://appassets.androidplatform.net.evil.example/assist",
                "https://evil.example/?host=appassets.androidplatform.net",
                "https://appassets.androidplatform.net@evil.example/assist",
                "https://evil.example@appassets.androidplatform.net/assist",
                "https://user:pass@appassets.androidplatform.net/assist",
                "https://appassets.androidplatform.net:444/assist",
                "https://appassets.androidplatform.net:0443/assist",
                "https://appassets.androidplatform.net:/assist",
                "https://appassets.androidplatform.net./assist",
                "https://%61ppassets.androidplatform.net/assist",
                "https://appassets.androidplatform.net\\@evil.example/assist",
                "https://appassets.androidplatform.net\n.evil.example/assist"}) {
            assertFalse(String.valueOf(url), BundlePolicy.isLocal(url));
            assertNull(String.valueOf(url), BundlePolicy.assetPath(url));
        }
    }

    @Test public void permitsOnlyTheBarcodeApiAtTheExactNetworkOrigin() {
        assertTrue(BundlePolicy.isAllowedNetwork(PRODUCT));
        assertTrue(BundlePolicy.isAllowedNetwork(PRODUCT + "?fields=product_name,allergens_tags"));
        assertTrue(BundlePolicy.isAllowedNetwork(PRODUCT.replace(".org/", ".org:443/")));
        for (String url : new String[] {
                null, "", PRODUCT.replace("https:", "http:"), PRODUCT.replace(".org/", ".org:8443/"),
                PRODUCT.replace("world.openfoodfacts.org", "world.openfoodfacts.org.evil.example"),
                PRODUCT.replace("world.openfoodfacts.org", "user@world.openfoodfacts.org"),
                PRODUCT.replace("world.openfoodfacts.org", "world.openfoodfacts.org@evil.example"),
                PRODUCT.replace("world.openfoodfacts.org", "world.openfoodfacts.org."),
                PRODUCT.replace("1234567890123", "123456789012"), PRODUCT.replace("1234567890123", "12345678901234"),
                PRODUCT.replace("1234567890123", "ABCDEFGHIJKLM"), PRODUCT.replace("1234567890123", "%31%32%33"),
                PRODUCT.replace(".json", ".json/extra"), PRODUCT.replace("/api/v2/", "/api/v2/../v2/"),
                PRODUCT.replace("/product/", "/product%2f"), "https://world.openfoodfacts.org/",
                "https://world.openfoodfacts.org/cgi/search.pl", LOCAL + "/api/v2/product/1234567890123.json"}) {
            assertFalse(String.valueOf(url), BundlePolicy.isAllowedNetwork(url));
        }
    }
}

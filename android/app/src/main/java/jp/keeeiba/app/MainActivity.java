package jp.keeeiba.app;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.CookieManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * 2強レース(個人用)。予想アプリ(GitHub Pages)を1画面で出す。
 * ブラウザの予想アプリからは netkeiba を読めないので、オッズをその場で取るときはアプリが代わりに取る(KeibaApp.fetch)。
 * AI競馬ナビの追切評価も同じようにアプリが取る(この端末で見るだけ。GitHub には保存しない)。
 */
public class MainActivity extends Activity {
    static final String APP_URL = "https://honda1986.github.io/keeeiba-app/keiba/?app=1";

    /** 裏に回っていたのがこれより長ければ、戻ったときに画面を読み直す(予想アプリの更新を出すため) */
    static final long RELOAD_AFTER_MS = 10 * 60 * 1000;

    WebView web;
    long pausedAt = 0;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        CookieManager.getInstance().setAcceptCookie(true);
        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);          // 予想アプリの localStorage(買い目の記録など)
        s.setDatabaseEnabled(true);
        s.setSupportMultipleWindows(false);
        web.setWebChromeClient(new WebChromeClient());
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                Uri u = req.getUrl();
                if ("honda1986.github.io".equals(u.getHost())) return false;
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, u));     // ほかのサイトはブラウザで開く
                } catch (Exception e) {
                    toast("開けません: " + u);
                }
                return true;
            }
        });
        web.addJavascriptInterface(new Bridge(), "KeibaApp");
        setContentView(web);
        loadFresh();                           // 前の画面は戻さない(古い画面のまま出さない)
    }

    /** 予想アプリを読み直す。GitHub Pages は10分キャッシュさせるので、毎回違うアドレスにして古い画面を出さない */
    void loadFresh() {
        web.loadUrl(APP_URL + "&v=" + System.currentTimeMillis());
    }

    void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }

    /** 予想アプリ(Web)から呼ばれる */
    class Bridge {
        /** netkeiba(race.netkeiba.com)・AI競馬ナビ(中央のレースのページ)を取って、文字を予想アプリに返す(window.kbFetched(id, 文字, エラー)) */
        @JavascriptInterface
        public void fetch(String url, String id) {
            if (url == null || !(url.startsWith("https://race.netkeiba.com/") || url.startsWith("https://horse-racing-ai-navi.com/jra/"))) {
                fetched(id, null, "このアドレスは取れません");
                return;
            }
            new Thread(() -> {
                String body = null, err = null;
                try {
                    HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
                    c.setConnectTimeout(10000);
                    c.setReadTimeout(15000);
                    c.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36");
                    int code = c.getResponseCode();
                    if (code >= 400) throw new Exception("HTTP " + code);
                    ByteArrayOutputStream b = new ByteArrayOutputStream();
                    try (InputStream in = c.getInputStream()) {
                        byte[] buf = new byte[8192];
                        int n;
                        while ((n = in.read(buf)) > 0) b.write(buf, 0, n);
                    }
                    // 文字コード: 返事に書いてあればそれ、無ければ netkeiba のページは EUC-JP、ほか(API・AI競馬ナビ)は UTF-8
                    String ct = c.getContentType() == null ? "" : c.getContentType();
                    String cs = ct.toLowerCase().contains("charset=") ? ct.substring(ct.toLowerCase().indexOf("charset=") + 8).trim()
                            : url.startsWith("https://race.netkeiba.com/") && !url.contains("/api/") ? "EUC-JP" : "UTF-8";
                    body = b.toString(cs.split(";")[0].replace("\"", "").trim());
                } catch (Exception e) {
                    err = e.toString();
                }
                fetched(id, body, err);
            }).start();
        }

        void fetched(String id, String body, String err) {
            String js = "window.kbFetched&&window.kbFetched(" + JSONObject.quote(id) + "," + (body == null ? "null" : JSONObject.quote(body))
                    + "," + JSONObject.quote(err == null ? "" : err) + ")";
            runOnUiThread(() -> web.evaluateJavascript(js, null));
        }

        @JavascriptInterface
        public String version() {
            try {
                return getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
            } catch (Exception e) {
                return "";
            }
        }
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onPause() {
        super.onPause();
        pausedAt = System.currentTimeMillis();
        CookieManager.getInstance().flush();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (pausedAt > 0 && System.currentTimeMillis() - pausedAt > RELOAD_AFTER_MS) loadFresh();
        pausedAt = 0;
    }
}

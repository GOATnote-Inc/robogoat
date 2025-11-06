# Brev Authentication Setup Guide

**Problem:** Tokens from `brev login --token ...` expire after 1-2 hours, requiring frequent re-authentication.

**Solution:** Use browser-based authentication for persistent sessions.

---

## **Recommended: Browser Authentication**

### **One-Time Setup:**

```bash
# On your Mac/local machine
brev login
```

This will:
1. Open your browser
2. Complete OAuth flow with NVIDIA SSO
3. Store refresh token in `~/.brev/.config.json`
4. Auto-renew access tokens (no more manual tokens!)

### **Connect to Instance:**

```bash
# Just works - no token needed
brev shell awesome-gpu-name
```

The refresh token automatically renews short-lived access tokens, so you never see "token expired" again.

---

## **For Automation: Service Account**

If you need long-lived tokens for CI/CD:

1. Contact [Brev Support](https://brev.dev/docs/cli)
2. Request service account token with 90-day TTL
3. Store in environment variable:

```bash
export BREV_TOKEN="your-long-lived-token"
echo "export BREV_TOKEN='your-long-lived-token'" >> ~/.bashrc
```

---

## **H100 Persistent Sessions**

Add to your H100's `~/.bashrc` for automatic credential refresh:

```bash
# Auto-refresh Brev credentials on login
if [ -f "$HOME/.brev/.config.json" ]; then
    export BREV_AUTH_TOKEN=$(jq -r '.access_token // empty' ~/.brev/.config.json 2>/dev/null)
    
    # Optional: Auto-refresh on shell start
    if command -v brev &> /dev/null; then
        brev login --refresh > /dev/null 2>&1 || true
    fi
fi
```

---

## **Troubleshooting**

### **"declined to login" Error**

**Cause:** Running `brev login` in non-interactive context (script, SSH)

**Fix:** Run manually in terminal:
```bash
brev login
# Press Enter when prompted
# Complete browser OAuth flow
```

### **Token Still Expiring**

**Cause:** Using old token-based auth method

**Fix:** Log out and log back in with browser auth:
```bash
brev logout
brev login  # Browser-based, not --token
```

### **Browser Doesn't Open**

**Cause:** SSH session or headless environment

**Fix:** Use token on local machine first:
```bash
# On local machine with browser
brev login

# Then SSH to remote
ssh user@remote
brev shell awesome-gpu-name  # Uses local credentials
```

---

## **Security Best Practices**

1. **Never commit tokens** - Use `.gitignore` or environment variables
2. **Rotate regularly** - Brev auto-handles rotation with browser auth
3. **Use Secure Links** - Restrict access to team members only
4. **Enable 2FA** - On your NVIDIA SSO account
5. **Log out when done** - `brev logout` if using shared machine

---

## **Quick Reference**

| Task | Command |
|------|---------|
| **Setup (once)** | `brev login` |
| **Connect to H100** | `brev shell awesome-gpu-name` |
| **Check status** | `brev ls` |
| **Logout** | `brev logout` |
| **Refresh** | `brev login --refresh` |

---

## **Why Browser Auth > Token Auth**

| Method | Duration | Auto-Refresh | Security |
|--------|----------|--------------|----------|
| `--token` | 1-2 hours | ❌ No | ⚠️ Token in bash history |
| Browser OAuth | 30 days | ✅ Yes | ✅ Secure, no exposed tokens |
| Service Account | 90 days | ❌ No | ✅ For automation only |

**Recommendation:** Use browser auth for daily work, service accounts only for CI/CD.

---

## **Next Steps**

1. Run `brev login` in your terminal (not in scripts)
2. Complete browser authentication
3. Connect with `brev shell awesome-gpu-name`
4. Never paste tokens again! 🎉

---

**Questions?** See [Brev CLI Documentation](https://brev.dev/docs/cli) or contact support.


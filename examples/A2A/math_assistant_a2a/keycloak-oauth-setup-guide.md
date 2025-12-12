<!-- SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

:::important
This guide is WIP and will be updated soon.
:::

# Keycloak OAuth2 Setup Guide for NAT A2A

This guide walks through setting up Keycloak as a proper OAuth2 authorization server for testing OAuth2-protected A2A servers in NAT.


## Prerequisites

- Docker installed and running
- NAT development environment set up
- No services running on port 8080

## Step 1: Start Keycloak

```bash
# Start Keycloak
docker run -d --name keycloak \
  -p 127.0.0.1:8080:8080 \
  -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin \
  quay.io/keycloak/keycloak:latest start-dev
```

**Wait for Keycloak to start** (about 30-60 seconds). Check logs:

```bash
docker logs -f keycloak
```

Look for: `Listening on: http://0.0.0.0:8080`

**Access Keycloak:** Open `http://localhost:8080` in your browser

## Step 2: Configure Keycloak Realm and Scopes

1. **Log in to Keycloak Admin Console:**
   - Username: `admin`
   - Password: `admin`

2. **Verify you're in the `master` realm** (top-left dropdown)

3. **Create the `calculator_a2a:execute` scope (for the calculator agent):**
   - Go to **Client scopes** (left sidebar)
   - Click **Create client scope**
   - Fill in:
     - **Name**: `calculator_a2a:execute`
     - **Type**: `Optional`
   - Click **Save**

4. **Verify OpenID Discovery endpoint:**
   ```bash
   curl http://localhost:8080/realms/master/.well-known/openid-configuration | python3 -m json.tool
   ```

   You should see:
   - `authorization_endpoint`
   - `token_endpoint`
   - `jwks_uri`
   - `registration_endpoint` (for DCR)

## Step 3: Register Math Assistant Client

You have two options:

### Option A: Manual Client Registration (Recommended for Testing)

1. In Keycloak Admin Console, go to **Clients** (left sidebar)
2. Click **Create client**
3. **General Settings:**
   - **Client ID**: `math-assistant-client`
   - **Client type**: `OpenID Connect`
   - Click **Next**

4. **Capability config:**
   - **Client authentication**: `On` (confidential client)
   - **Authorization**: `Off`
   - **Authentication flow:**
     - ✓ Standard flow (authorization code)
     - ✓ Direct access grants
   - Click **Next**

5. **Login settings:**
   - **Valid redirect URIs**: `http://localhost:8000/auth/redirect`
   - **Web origins**: `http://localhost:8000`
   - Click **Save**

6. **Get client credentials:**
   - Go to **Credentials** tab
   - Copy the **Client secret**
   - Note the **Client ID**: `math-assistant-client`

7. **Configure client scopes:**
   - Go to **Client scopes** tab
   - Click **Add client scope**
   - Select `calculator_a2a:execute`
   - Choose **Optional**
   - Click **Add**

### Option B: Dynamic Client Registration (DCR)

NAT's OAuth2 provider can use DCR if Keycloak is configured to allow it. By default, Keycloak restricts anonymous DCR. To enable it:

1. Go to **Realm settings** > **Client registration**
2. Click **Client registration policies** tab
3. Configure anonymous access or trusted hosts

**Note:** For testing, manual registration (Option A) is simpler.

## Step 4: Set Environment Variables

After registering the client:

```bash
# Set these in your terminal where you'll run the NAT client
export CALCULATOR_CLIENT_ID="math-assistant-client"
export CALCULATOR_CLIENT_SECRET="<paste-client-secret-from-keycloak>"

# Verify they're set
echo "Client ID: ${CALCULATOR_CLIENT_ID}"
echo "Client Secret: ${CALCULATOR_CLIENT_SECRET:0:10}..."
```

## Step 5: Start the Protected Calculator Server

```bash
# Terminal 1
nat a2a serve --config_file examples/A2A/calculator_a2a/configs/config-protected-oauth2.yml
```

You should see:
```
[INFO] OAuth2 token validation enabled for A2A server
[INFO] Starting A2A server 'Protected Calculator' at http://localhost:10000
```

## Step 6: Run the Math Assistant Client

```bash
# Terminal 2
# Make sure environment variables are set
export CALCULATOR_CLIENT_ID="math-assistant-client"
export CALCULATOR_CLIENT_SECRET="<your-client-secret>"

nat run --config_file examples/A2A/math_assistant_a2a/configs/config-a2a-auth-calc.yml \
  --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

**What should happen:**

1. **Browser opens** with Keycloak login page
2. **Log in** with any user (or create one)
3. **Consent page** shows requesting `calculator_a2a:execute` scope - click **Yes**
4. **Browser redirects** back to `localhost:8000/auth/redirect`
5. **Workflow continues** and calls the calculator
6. **Response returned** successfully

## Verification and Testing

### Verify JWKS Endpoint Works

```bash
curl http://localhost:8080/realms/master/protocol/openid-connect/certs | python3 -m json.tool
```

You should see public keys in JSON format.

### Test Without Authentication (Should Fail)

```bash
curl -X POST http://localhost:10000/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'
```

Expected: `401 Unauthorized` or `403 Forbidden`

### Check Token Contents

After a successful OAuth flow, you can decode the JWT token at [jwt.io](https://jwt.io) to see:
- `iss`: Should be `http://localhost:8080/realms/master`
- `aud`: Should include your server URL
- `scope` or `scp`: Should include `calculator_a2a:execute`
- `exp`: Expiration timestamp
- `sub`: User subject/ID

## Troubleshooting

### "Token is not active" Error

**Cause:** Token validation failing on the resource server.

**Check:**
1. JWKS URI is correct and accessible
2. Token `iss` matches the `issuer_url` in server config
3. Token `aud` matches the `audience` in server config (if set)
4. Token hasn't expired
5. Clock skew isn't too large between client/server

### "invalid_client" Error

**Cause:** Client credentials are wrong or client isn't registered.

**Fix:**
1. Verify client exists in Keycloak
2. Check `CALCULATOR_CLIENT_ID` and `CALCULATOR_CLIENT_SECRET` are set correctly
3. Make sure redirect URI matches exactly

### "invalid_scope" Error

**Cause:** Requested scope not allowed for client.

**Fix:**
1. Go to Keycloak → Clients → `math-assistant-client` → Client scopes
2. Make sure `calculator_a2a:execute` is added as an optional or default scope

### Browser Doesn't Open

**Cause:** NAT can't detect or open the browser.

**Workaround:**
1. Look for the authorization URL in the console output
2. Copy and paste it into your browser manually

## Cleanup

To stop and remove Keycloak:

```bash
docker stop keycloak
docker rm keycloak
```

To restart with clean state:

```bash
docker rm -f keycloak
# Then run the start command again
```

/**
 * SafeFall Cloud Functions
 * Privileged Operations: Incident Notification Dispatch, Family Invitation Redemption,
 * Device Pairing Verification, and Authorized Incident Lifecycle Management.
 */

const functions = require("firebase-functions");
const admin = require("firebase-admin");

admin.initializeApp();
const db = admin.firestore();
const messaging = admin.messaging();

/**
 * Triggered whenever a new incident is created or published.
 * Fan-out push notifications via Firebase Cloud Messaging to all active family caregivers & admins.
 */
exports.onIncidentPublished = functions.firestore
  .document("incidents/{incidentId}")
  .onCreate(async (snap, context) => {
    const incidentId = context.params.incidentId;
    const data = snap.data();

    if (data.incidentStatus !== "EMERGENCY_PUBLISHED") {
      console.log(`Incident ${incidentId} created with status ${data.incidentStatus}. Skipping emergency alert.`);
      return null;
    }

    const familyId = data.familyId;
    console.log(`Processing Emergency Fall Incident ${incidentId} for family ${familyId}`);

    try {
      // 1. Fetch all active memberships for this family
      const membersSnap = await db
        .collection("memberships")
        .where("familyId", "==", familyId)
        .where("membershipStatus", "==", "active")
        .get();

      if (membersSnap.empty) {
        console.warn(`No active members found for family ${familyId}.`);
        return null;
      }

      const recipientUids = [];
      membersSnap.forEach((doc) => {
        const m = doc.data();
        if (m.notificationPreferences?.pushEnabled !== false) {
          recipientUids.push(m.uid);
        }
      });

      if (recipientUids.length === 0) {
        console.log("No members with pushEnabled = true.");
        return null;
      }

      // 2. Collect FCM tokens for these UIDs
      const tokens = [];
      const userRefs = recipientUids.map((uid) => db.collection("users").doc(uid));
      const userDocs = await db.getAll(...userRefs);

      userDocs.forEach((doc) => {
        if (doc.exists) {
          const u = doc.data();
          if (Array.isArray(u.fcmTokens)) {
            tokens.push(...u.fcmTokens);
          }
        }
      });

      // Deduplicate tokens
      const uniqueTokens = [...new Set(tokens)];
      console.log(`Sending FCM notifications to ${uniqueTokens.length} device tokens.`);

      const title = `🚨 SAFE FALL ALERT: ${data.deviceName || "Monitor"} at ${data.locationLabel || "Home"}`;
      const body = `Suspected fall detected (${data.transitionType || "FALL"}). Voice response: ${data.voiceResponseStatus || "NO_RESPONSE"}. Tap to inspect immediately.`;

      // 3. Dispatch multicast notification (No private evidence images in push payload per privacy spec)
      if (uniqueTokens.length > 0) {
        const messagePayload = {
          tokens: uniqueTokens,
          notification: {
            title: title,
            body: body,
          },
          data: {
            incidentId: incidentId,
            familyId: familyId,
            detectedAt: data.detectedAt || new Date().toISOString(),
            status: data.incidentStatus,
            deepLink: `safefall://incident/${incidentId}`,
          },
          android: {
            priority: "high",
            notification: {
              channelId: "safefall_emergency_channel",
              sound: "emergency_alarm",
              priority: "max",
            },
          },
          apns: {
            payload: {
              aps: {
                sound: "emergency_alarm.caf",
                badge: 1,
                "interruption-level": "critical",
              },
            },
          },
        };

        const response = await messaging.sendEachForMulticast(messagePayload);
        console.log(`Multicast sent: ${response.successCount} succeeded, ${response.failureCount} failed.`);

        // Record notification audit
        await db.collection("notifications").add({
          incidentId: incidentId,
          familyId: familyId,
          recipientCount: uniqueTokens.length,
          successCount: response.successCount,
          failureCount: response.failureCount,
          createdAt: admin.firestore.FieldValue.serverTimestamp(),
        });
      }

      // 4. Record event in incidentEvents
      await db.collection("incidentEvents").add({
        incidentId: incidentId,
        eventType: "NOTIFICATIONS_DISPATCHED",
        timestamp: admin.firestore.FieldValue.serverTimestamp(),
        metadata: {
          recipients: recipientUids.length,
          tokensCount: uniqueTokens.length,
        },
      });

      return { success: true };
    } catch (err) {
      console.error("Error in onIncidentPublished:", err);
      return null;
    }
  });

/**
 * Callable Function: Redeem Family Invitation Code
 * Secure validation of expiration, usage count, and atomic role assignment.
 */
exports.redeemInvitation = functions.https.onCall(async (data, context) => {
  if (!context.auth) {
    throw new functions.https.HttpsError("unauthenticated", "User must be authenticated to join a family.");
  }

  const { inviteCode } = data;
  const uid = context.auth.uid;

  if (!inviteCode || typeof inviteCode !== "string") {
    throw new functions.https.HttpsError("invalid-argument", "Valid invitation code required.");
  }

  const cleanCode = inviteCode.trim().toUpperCase();

  return db.runTransaction(async (transaction) => {
    // 1. Query matching invitation
    const invSnap = await transaction.get(
      db.collection("invitations").where("inviteCode", "==", cleanCode).limit(1)
    );

    if (invSnap.empty) {
      throw new functions.https.HttpsError("not-found", "Invalid invitation code.");
    }

    const invDoc = invSnap.docs[0];
    const inv = invDoc.data();

    // 2. Validate expiration
    if (inv.expiresAt && new Date(inv.expiresAt) < new Date()) {
      throw new functions.https.HttpsError("failed-precondition", "This invitation code has expired.");
    }

    // 3. Validate usage limit
    if (inv.usageLimit && inv.usageCount >= inv.usageLimit) {
      throw new functions.https.HttpsError("failed-precondition", "This invitation code has reached its maximum usage limit.");
    }

    if (inv.revokedAt) {
      throw new functions.https.HttpsError("failed-precondition", "This invitation has been revoked.");
    }

    // 4. Create or update membership
    const membershipId = `${inv.familyId}_${uid}`;
    const membershipRef = db.collection("memberships").doc(membershipId);

    transaction.set(membershipRef, {
      membershipId: membershipId,
      familyId: inv.familyId,
      familyName: inv.familyName || "Family Residence",
      uid: uid,
      role: inv.role || "caregiver",
      joinedAt: admin.firestore.FieldValue.serverTimestamp(),
      membershipStatus: "active",
      notificationPreferences: {
        pushEnabled: true,
        soundEnabled: true,
        criticalAlerts: true,
      },
    });

    // 5. Increment usage count
    transaction.update(invDoc.ref, {
      usageCount: admin.firestore.FieldValue.increment(1),
    });

    return {
      success: true,
      familyId: inv.familyId,
      familyName: inv.familyName,
      role: inv.role || "caregiver",
    };
  });
});

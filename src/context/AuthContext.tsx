import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { onAuthStateChanged, sendPasswordResetEmail, signInWithEmailAndPassword, signOut, type User } from "firebase/auth";
import { firebaseAuth } from "../lib/firebase";
import {
  ensureUserProfile,
  getUserRoleFromProfile,
  registerUserLogin,
  registerUserLogout,
  syncMyRoleClaim,
  type AppUserRole,
} from "../services/users";

type AuthContextType = {
  user: User | null;
  appRole: AppUserRole | null;
  isAuthReady: boolean;
  isAuthenticated: boolean;
  isSecondFactorVerified: boolean;
  login: (email: string, password: string) => Promise<{ ok: boolean; message?: string }>;
  sendResetPassword: (email: string) => Promise<{ ok: boolean; message?: string }>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [appRole, setAppRole] = useState<AppUserRole | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);

  async function hydrateUserSecurity(nextUser: User) {
    try {
      await ensureUserProfile(nextUser);
      await syncMyRoleClaim();
      await nextUser.getIdToken(true);
      const tokenResult = await nextUser.getIdTokenResult();
      const role = tokenResult.claims.appRole;
      setAppRole(
        role === "Admin" || role === "Supervisor" || role === "Operador"
          ? (role as AppUserRole)
          : "Operador",
      );
    } catch (error) {
      // No bloquea sesion autenticada por fallos operativos secundarios.
      console.error("No fue posible sincronizar el perfil de seguridad:", error);
      try {
        const profileRole = await getUserRoleFromProfile(nextUser.uid);
        setAppRole(profileRole ?? "Operador");
      } catch {
        setAppRole((prev) => prev ?? "Operador");
      }
    }
  }

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(firebaseAuth, async (nextUser) => {
      if (nextUser) {
        await hydrateUserSecurity(nextUser);
      } else {
        setAppRole(null);
      }
      setUser(nextUser);
      setIsAuthReady(true);
    });

    return () => unsubscribe();
  }, []);

  async function login(email: string, password: string) {
    try {
      const credentials = await signInWithEmailAndPassword(firebaseAuth, email, password);
      try {
        await registerUserLogin(credentials.user);
        await hydrateUserSecurity(credentials.user);
      } catch (error) {
        // Login ya fue exitoso; este error no debe bloquear acceso.
        console.error("Login exitoso, pero fallo una sincronizacion posterior:", error);
      }
      return { ok: true };
    } catch (error) {
      const code = typeof error === "object" && error && "code" in error ? String(error.code) : "";

      if (code === "auth/invalid-credential" || code === "auth/wrong-password" || code === "auth/user-not-found") {
        return { ok: false, message: "Credenciales incorrectas" };
      }

      if (code === "auth/too-many-requests") {
        return { ok: false, message: "Demasiados intentos. Intenta nuevamente en unos minutos." };
      }

      return { ok: false, message: "No fue posible iniciar sesion. Verifica la configuracion de Firebase." };
    }
  }

  async function logout() {
    if (firebaseAuth.currentUser) {
      try {
        await registerUserLogout(firebaseAuth.currentUser);
      } catch (error) {
        // No debe bloquear el cierre de sesion si falla el registro operativo.
        console.error("No fue posible registrar el logout del usuario:", error);
      }
    }
    await signOut(firebaseAuth);
    setAppRole(null);
  }

  async function sendResetPassword(email: string) {
    try {
      await sendPasswordResetEmail(firebaseAuth, email);
      return { ok: true, message: "Se envio un enlace de recuperacion al correo indicado." };
    } catch (error) {
      const code = typeof error === "object" && error && "code" in error ? String(error.code) : "";

      if (code === "auth/missing-email" || !email.trim()) {
        return { ok: false, message: "Ingresa un correo valido para recuperar la cuenta." };
      }

      if (code === "auth/invalid-email") {
        return { ok: false, message: "El correo no tiene un formato valido." };
      }

      return { ok: false, message: "No fue posible enviar el correo de recuperacion." };
    }
  }

  const value = useMemo(
    () => ({
      user,
      appRole,
      isAuthReady,
      isAuthenticated: Boolean(user),
      isSecondFactorVerified: Boolean(user),
      login,
      sendResetPassword,
      logout,
    }),
    [appRole, isAuthReady, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}

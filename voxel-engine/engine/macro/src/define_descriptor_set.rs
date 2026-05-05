use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr, Ident, Token, Visibility};

struct DefineDescriptorSetInput {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    lifetime: Ident,
    pool: Ident,
    config: Expr,
}

impl Parse for DefineDescriptorSetInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let ident: Ident = input.parse()?;
        assert_eq!(ident, "lifetime", "Expected 'lifetime'");
        input.parse::<Token![=]>()?;
        let lifetime: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let ident: Ident = input.parse()?;
        assert_eq!(ident, "pool", "Expected 'pool'");
        input.parse::<Token![=]>()?;
        let pool: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let ident: Ident = input.parse()?;
        assert_eq!(ident, "config", "Expected 'config'");
        input.parse::<Token![=]>()?;
        let config: Expr = input.parse()?;
        input.parse::<Token![;]>()?;

        Ok(DefineDescriptorSetInput {
            attrs,
            vis,
            name,
            lifetime,
            pool,
            config,
        })
    }
}

pub fn define_descriptor_set_impl(input: TokenStream) -> TokenStream {
    let DefineDescriptorSetInput {
        attrs,
        vis,
        name,
        lifetime,
        pool,
        config,
    } = syn::parse_macro_input!(input as DefineDescriptorSetInput);

    let lifetime_str = lifetime.to_string();
    let ids_total_count: TokenStream2 = match lifetime_str.as_str() {
        "Persistent" => quote! { 1 },
        "PerFrame" => quote! { crate::vulkan::vulkan_renderer_core::MAX_FRAMES_IN_FLIGHT },
        _ => {
            return syn::Error::new(
                lifetime.span(),
                "Expected 'Persistent' or 'PerFrame' for lifetime",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name;

        impl crate::vulkan::resource_config::DescriptorSetMarker for #name {
            type Lifetime = crate::vulkan::resource_lifetime::#lifetime;
            type Pool = #pool;
            const CONFIG: crate::vulkan::resource_config::DescriptorSetConfig = #config;

            fn ids_slice() -> &'static [std::sync::atomic::AtomicUsize] {
                static IDS: [std::sync::atomic::AtomicUsize; #ids_total_count] = unsafe {
                    std::mem::transmute([usize::MAX; #ids_total_count])
                };
                &IDS
            }
        }
    };

    expanded.into()
}
